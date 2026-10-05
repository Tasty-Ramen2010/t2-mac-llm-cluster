"""IDE plugin for agent_server.py: turns the chat agent into a small coding team.

The planner (the big model behind the model port, e.g. Qwen3.6-35B-A3B on the AGX) gets two extra tools:

  delegate_code  hand a self-contained coding task to the fast "operative" models (Maple replicas behind the router),
                 optionally writing the result straight into the workspace so the planner does not have to re-type it.
  git_clone      clone a (private) GitHub repo into the workspace with a read-only token that lives only on the server.

Everything else (sandbox, files, permissions, chats) is the existing agent. Enabled by two lines at the end of
agent_server.py; remove them (or this file) and the agent behaves exactly as before.

Config (env or ~/.config/ide/): IDE_WORKER_URL (default http://127.0.0.1:8095/v1/chat/completions),
~/.config/ide/github_token (chmod 600, fine-grained read-only token).
"""
import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

WORKER_URL = os.environ.get("IDE_WORKER_URL", "http://127.0.0.1:8095/v1/chat/completions")
CONF = Path(os.environ.get("IDE_HOME", str(Path.home() / ".config" / "ide")))
TOKEN_FILE = CONF / "github_token"
MAX_CONTEXT_CHARS = 24000
REPO_RE = re.compile(r"^[A-Za-z0-9_.-]{1,100}/[A-Za-z0-9_.-]{1,100}$")
REF_RE = re.compile(r"^[A-Za-z0-9_./-]{1,100}$")
DEST_RE = re.compile(r"^[A-Za-z0-9_.-][A-Za-z0-9_./-]{0,100}$")

IDE_TOOLS = [
    {"type": "function", "function": {
        "name": "delegate_code",
        "description": "Hand ONE self-contained coding task to a fast worker model (Maple) and get its code back. Use it for "
                       "writing a function, class, module or test file from a precise spec; you review and test the result. "
                       "Give: what to build, the exact file name, inputs/outputs, constraints, and any existing code it must "
                       "match (list those files in 'files'). With 'write_to' the worker's code is saved straight into that "
                       "workspace file and you only get a short summary back, so call read_file if you need the whole thing. "
                       "Keep each task small (about 150 lines or less) and independent of other tasks.",
        "parameters": {"type": "object", "properties": {
            "task": {"type": "string", "description": "precise spec of the code to write"},
            "files": {"type": "array", "items": {"type": "string"},
                      "description": "workspace files the worker should read first (context)"},
            "write_to": {"type": "string", "description": "workspace path to save the code to, e.g. src/parser.py"},
            "language": {"type": "string", "description": "e.g. python, c, javascript (optional)"}},
            "required": ["task"]}}},
    {"type": "function", "function": {
        "name": "git_clone",
        "description": "Clone a GitHub repository (private ones work) into this conversation's workspace. Give 'owner/name'. "
                       "The clone is shallow by default. Afterwards use list_files / read_file / run_shell on the new folder.",
        "parameters": {"type": "object", "properties": {
            "repo": {"type": "string", "description": "owner/name, e.g. octocat/hello-world"},
            "ref": {"type": "string", "description": "branch or tag (optional)"},
            "dest": {"type": "string", "description": "folder name in the workspace (default: the repo name)"}},
            "required": ["repo"]}}},
]

IDE_TOOLS += [
    {"type": "function", "function": {
        "name": "run_on_agx",
        "description": "Run a bash command on the AGX Orin GPU computer (CUDA 11.4 with nvcc, gcc, cmake, Python 3.8, sm_87 GPU). Your "
                       "workspace folder (or 'dir' inside it) is copied there, the command runs as an unprivileged user with NO "
                       "network access and a hard time limit, and the files it creates or changes are copied back. Use it to "
                       "compile and run CUDA/GPU code and benchmarks. Max 30 minutes. GPU memory is shared with the model server "
                       "(about 3 GB is free), so keep GPU allocations under ~2 GB; jobs that use too much memory are killed.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"},
            "dir": {"type": "string", "description": "workspace sub-folder to run in (default: the whole workspace, max 50 MB)"},
            "timeout": {"type": "integer", "description": "seconds, default 120, max 1800"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "git_push",
        "description": "Commit all changes in a git repository in the workspace and push them to a BRANCH of the owner's GitHub repo. "
                       "Pushing to main/master/develop/release branches is refused; pick a feature branch name. Only repositories owned "
                       "by the configured GitHub owner(s) are allowed. Use github_pr afterwards to open a pull request.",
        "parameters": {"type": "object", "properties": {
            "dir": {"type": "string", "description": "repository folder in the workspace (default .)"},
            "branch": {"type": "string", "description": "branch to push to, e.g. agent/fix-parser"},
            "message": {"type": "string", "description": "commit message"}}, "required": ["branch", "message"]}}},
    {"type": "function", "function": {
        "name": "github_pr",
        "description": "Open a pull request on one of the owner's GitHub repositories from a branch you pushed with git_push.",
        "parameters": {"type": "object", "properties": {
            "repo": {"type": "string", "description": "owner/name"}, "head": {"type": "string", "description": "the pushed branch"},
            "title": {"type": "string"}, "body": {"type": "string"},
            "base": {"type": "string", "description": "target branch (default: the repo default branch)"}}, "required": ["repo", "head", "title"]}}},
]

IDE_SYSTEM = (
    " You are the lead engineer of a small coding team. For anything beyond a few lines, split the work into small, "
    "independent pieces and use delegate_code to have fast worker models write them (give each a precise spec and a "
    "write_to file), then read their files, run the code or tests, and fix problems yourself or send a corrected spec. "
    "Do tiny edits yourself. To work on a GitHub repository (including private ones the owner has authorised) use "
    "git_clone with owner/name, then read its README and structure before changing anything. Always run the code "
    "you or the workers wrote before saying it works. Your sandbox has OPEN internet access (public websites only: pip, npm, git, curl, "
    "Hugging Face; no permission pop-ups), commands may run up to 30 minutes, and you may start servers inside the sandbox on "
    "127.0.0.1 ports 30000-30999 (run them in the background with nohup ... &, then test with curl; stray processes are cleaned up "
    "after about 40 minutes). For GPU/CUDA work use run_on_agx (nvcc is there; no network, files are synced back). To save work "
    "to GitHub use git_push to a feature branch (never main) and github_pr to open a pull request. "
    "Ignore older instructions that say only ports 80/443 exist or that there is no GPU."
)


def _clip(text, n=1800):
    text = text or ""
    return text if len(text) <= n else text[: n // 2] + "\n...[clipped]...\n" + text[-n // 2:]


# ---------- delegate_code ----------

def _fenced(text):
    m = re.search(r"```[A-Za-z0-9_+.-]*\n(.*?)```", text, re.S)
    return m.group(1).rstrip("\n") + "\n" if m else None


def delegate_code(A, args, chat_id):
    task = (args.get("task") or "").strip()
    if not task:
        return "delegate_code: 'task' is empty."
    context, used = [], 0
    for path in args.get("files") or []:
        body = A.read_file(path, chat_id)
        body = body if isinstance(body, str) else str(body)
        chunk = body[: max(0, min(MAX_CONTEXT_CHARS - used, 12000))]
        used += len(chunk)
        context.append(f"--- file: {path} ---\n{chunk}")
    lang = args.get("language") or ""
    user = task + ("\n\nExisting files for reference:\n" + "\n".join(context) if context else "")
    messages = [
        {"role": "system", "content": "You are a fast, careful coding assistant working for a lead engineer. Reply with the "
                                      "complete requested code in ONE fenced code block" + (f" ({lang})" if lang else "") +
                                      ", then at most three short bullet notes. No introduction, no repeating the task."},
        {"role": "user", "content": user},
    ]
    body = {"messages": messages, "max_tokens": 3500, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0,
            "presence_penalty": 1.0, "reasoning_budget_tokens": 1024,
            "reasoning_budget_message": "\n\nI've thought enough; now I'll write the code.\n"}
    t0 = time.time()
    try:
        req = urllib.request.Request(WORKER_URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.load(r)
    except (urllib.error.URLError, OSError, ValueError) as e:
        return f"delegate_code: worker unavailable ({e}). Write this code yourself instead."
    msg = (data.get("choices") or [{}])[0].get("message", {})
    text = (msg.get("content") or "").strip()
    secs = time.time() - t0
    toks = (data.get("usage") or {}).get("completion_tokens", 0)
    tps = (data.get("timings") or {}).get("predicted_per_second") or (toks / secs if secs else 0)
    tag = f"[worker: {toks} tokens, {secs:.0f}s, {tps:.0f} tok/s]"
    code = _fenced(text)
    dest = args.get("write_to")
    if dest and code:
        res = A.write_file(dest, code, chat_id)
        lines = code.count("\n")
        head = "\n".join(code.splitlines()[:25])
        notes = text.split("```")[-1].strip() if text.count("```") >= 2 else ""
        return f"{tag} wrote {lines} lines to {dest} ({res}).\n--- first lines ---\n{head}\n" + (f"--- worker notes ---\n{_clip(notes, 600)}" if notes else "")
    return f"{tag}\n{_clip(text, 6000)}"


# ---------- git_clone ----------

def _token():
    """Server-side GitHub token: ~/.config/ide/github_token if present, else this machine's `gh` login. Never shown to the model."""
    try:
        return TOKEN_FILE.read_text().strip()
    except OSError:
        pass
    try:
        p = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=10)
        if p.returncode == 0:
            return p.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return ""


def git_clone(A, args, chat_id):
    repo = (args.get("repo") or "").strip()
    repo = re.sub(r"^https://github\.com/", "", repo).removesuffix(".git").strip("/")
    if not REPO_RE.match(repo):
        return "git_clone: give the repository as owner/name (GitHub only)."
    ref = (args.get("ref") or "").strip()
    if ref and not REF_RE.match(ref):
        return "git_clone: invalid branch/tag name."
    dest = (args.get("dest") or repo.split("/")[1]).strip()
    if not DEST_RE.match(dest) or ".." in dest:
        return "git_clone: invalid destination folder."
    ws = A.workspace(chat_id or "scratch")
    token = _token()
    cmd = ["git"]
    if token:                               # header is passed on this one command line only; never stored in .git/config
        basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        cmd += ["-c", f"http.https://github.com/.extraheader=AUTHORIZATION: basic {basic}"]
    cmd += ["clone", "--depth", "50", "--no-tags"] + (["--branch", ref] if ref else []) + [f"https://github.com/{repo}.git", "repo"]
    tmp = tempfile.mkdtemp(prefix="ide-clone-")
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "/bin/true"}
    try:
        p = subprocess.run(cmd, cwd=tmp, env=env, capture_output=True, text=True, timeout=300)
        if p.returncode != 0:
            err = re.sub(r"basic [A-Za-z0-9+/=]+", "basic ***", p.stderr)
            hint = "" if token else " (no GitHub token is configured on the server, so only public repos work)"
            return f"git clone failed: {_clip(err, 600)}{hint}"
        src = Path(tmp) / "repo"
        target = Path(str(ws)) / dest
        # the workspace belongs to the sandbox user: move the clone in as that user
        subprocess.run(["sudo", "-n", "chown", "-R", "llmtools:llmtools", str(src)], check=True, capture_output=True, timeout=60)
        subprocess.run(["sudo", "-n", "rm", "-rf", str(target)], check=True, capture_output=True, timeout=60)
        subprocess.run(["sudo", "-n", "mv", str(src), str(target)], check=True, capture_output=True, timeout=60)
        n = sum(1 for _ in subprocess.run(["sudo", "-n", "find", str(target), "-type", "f", "-not", "-path", "*/.git/*"],
                                          capture_output=True, text=True, timeout=60).stdout.splitlines())
        return f"Cloned {repo}{' @ ' + ref if ref else ''} into ./{dest} ({n} files). Use list_files / read_file to explore it."
    except subprocess.TimeoutExpired:
        return "git clone timed out."
    except subprocess.CalledProcessError as e:
        return f"git_clone: could not place the clone in the workspace ({e})."
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------- show the planner as "the loaded model" (the stock collector only knows local llama-server processes) ----------

_MI = {"t": 0.0, "v": None}


def planner_info(A):
    now = time.time()
    if _MI["v"] and now - _MI["t"] < 10:
        return _MI["v"]
    try:
        base = A.LLM.rsplit("/v1/", 1)[0]
        with urllib.request.urlopen(base + "/props", timeout=3) as r:
            props = json.load(r)
        gen = props.get("default_generation_settings") or {}
        v = {"up": True, "name": "Qwen3.6-35B-A3B on the AGX", "kind": "MoE 35B, 3B active, GPU + MTP",
             "file": os.path.basename(props.get("model_path") or "planner.gguf"), "n_ctx": gen.get("n_ctx") or 0}
    except (OSError, ValueError, urllib.error.URLError):
        return None
    _MI["t"], _MI["v"] = now, v
    return v


# ---------- more powers: AGX GPU jobs, GitHub push/PR, process cleanup ----------

import threading
import uuid

AGX_SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "StrictHostKeyChecking=accept-new", "-p", "2223",
           os.environ.get("IDE_AGX", "root@10.10.13.1")]
AGX_JOB = "sh /mnt/persistent/data/agx-setup/agx-job"
AGX_LOCK = threading.Lock()
POLICY_FILE = CONF / "policy.json"
PROTECTED_BRANCHES = {"main", "master", "develop", "dev", "production", "prod", "trunk"}


def _policy():
    try:
        return json.loads(POLICY_FILE.read_text())
    except (OSError, ValueError):
        return {}


def _gh_user():
    try:
        p = subprocess.run(["gh", "api", "user", "-q", ".login"], capture_output=True, text=True, timeout=15)
        return p.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _allowed_owner(owner):
    owners = _policy().get("allowed_owners") or [_gh_user()]
    return owner.lower() in [o.lower() for o in owners if o]


def run_on_agx(A, args, chat_id):
    cmd = (args.get("command") or "").strip()
    if not cmd:
        return "run_on_agx: 'command' is empty."
    timeout = max(5, min(int(args.get("timeout") or 120), 1800))
    ws = Path(str(A.workspace(chat_id or "scratch")))
    sub = (args.get("dir") or ".").strip()
    if ".." in Path(sub).parts or Path(sub).is_absolute():
        return "run_on_agx: dir must be a folder inside the workspace."
    src = ws / sub
    job = "j" + uuid.uuid4().hex[:10]
    if not AGX_LOCK.acquire(timeout=30):
        return "run_on_agx: another GPU job is running; try again in a minute."
    t0 = time.time()
    try:
        size = subprocess.run(["sudo", "-n", "du", "-sk", str(src)], capture_output=True, text=True, timeout=30).stdout.split()
        if not size:
            return f"run_on_agx: folder {sub} not found in the workspace."
        if int(size[0]) > 50 * 1024:
            return f"run_on_agx: {sub} is {int(size[0]) // 1024} MB; pass a smaller 'dir' (max 50 MB)."
        up = subprocess.run(f"sudo -n tar -c --exclude=.git -C '{src}' . | " + " ".join(AGX_SSH) + f" '{AGX_JOB} put {job}'",
                            shell=True, capture_output=True, text=True, timeout=120)
        if "put ok" not in up.stdout:
            return f"run_on_agx: could not reach the AGX ({(up.stderr or up.stdout)[-200:].strip()})"
        enc = base64.b64encode(cmd.encode())
        run = subprocess.run(AGX_SSH + [f"{AGX_JOB} run {job} {timeout}"], input=enc, capture_output=True, timeout=timeout + 60)
        out = run.stdout.decode("utf-8", "replace")
        # bring changed files back into the workspace (as the sandbox user, size-capped on the AGX side)
        tarf = tempfile.mktemp(prefix="agxget-")
        with open(tarf, "wb") as f:
            subprocess.run(AGX_SSH + [f"{AGX_JOB} get {job}"], stdout=f, stderr=subprocess.DEVNULL, timeout=180)
        back = subprocess.run(["sudo", "-n", "-u", "llmtools", "tar", "-x", "-C", str(src), "--no-same-owner", "--no-absolute-names",
                               "-f", tarf, "--keep-newer-files"], capture_output=True, text=True, timeout=120)
        os.unlink(tarf)
        subprocess.run(AGX_SSH + [f"{AGX_JOB} rm {job}"], capture_output=True, timeout=30)
        note = "" if back.returncode == 0 else f" (copy-back warning: {back.stderr[-120:].strip()})"
        return f"{_clip(out, 7000).rstrip()}\n[AGX job {time.time() - t0:.0f}s, files synced back to ./{sub}]{note}"
    except subprocess.TimeoutExpired:
        return "run_on_agx: timed out."
    except Exception as e:
        return f"run_on_agx failed: {e}"
    finally:
        AGX_LOCK.release()


def _repo_of(gitdir):
    p = subprocess.run(["sudo", "-n", "-u", "llmtools", "git", "-c", "safe.directory=*", "-C", str(gitdir), "remote", "get-url", "origin"],
                       capture_output=True, text=True, timeout=30)
    m = re.search(r"github\.com[:/]([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/?$", p.stdout.strip())
    return (m.group(1), m.group(2)) if m else (None, None)


def git_push(A, args, chat_id):
    branch = (args.get("branch") or "").strip()
    msg = (args.get("message") or "").strip()
    if not REF_RE.match(branch) or not msg:
        return "git_push: give a valid branch name and a commit message."
    if branch.lower() in PROTECTED_BRANCHES or branch.lower().startswith("release") or branch.lower().startswith("refs/"):
        return f"git_push: pushing to '{branch}' is not allowed. Use a feature branch (for example agent/{re.sub('[^A-Za-z0-9._-]+', '-', msg.lower())[:30]}) and open a pull request."
    ws = Path(str(A.workspace(chat_id or "scratch")))
    sub = (args.get("dir") or ".").strip()
    if ".." in Path(sub).parts or Path(sub).is_absolute():
        return "git_push: dir must be inside the workspace."
    d = ws / sub
    owner, name = _repo_of(d)
    if not owner:
        return "git_push: that folder is not a git repository with a github.com 'origin' remote (use git_clone first)."
    if not _allowed_owner(owner):
        return f"git_push: pushing to {owner}/{name} is not allowed (only repositories owned by the configured GitHub owner)."
    token = _token()
    if not token:
        return "git_push: no GitHub login is configured on the server."
    login = _gh_user() or owner
    ident = ["-c", f"user.name=Cluster IDE agent", "-c", f"user.email={login}@users.noreply.github.com", "-c", "safe.directory=*"]
    llm = ["sudo", "-n", "-u", "llmtools", "env", "HOME=/home/llmtools", "git"] + ident + ["-C", str(d)]
    subprocess.run(llm + ["add", "-A"], capture_output=True, text=True, timeout=120)
    c = subprocess.run(llm + ["commit", "-m", msg + "\n\nAuthored by the cluster IDE agent (local Qwen3.6-35B-A3B + Maple workers)."],
                       capture_output=True, text=True, timeout=120)
    if c.returncode != 0 and "nothing to commit" not in (c.stdout + c.stderr):
        return f"git_push: commit failed: {_clip(c.stdout + c.stderr, 400)}"
    basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    p = subprocess.run(["git", "-c", "safe.directory=*", "-c", f"http.https://github.com/.extraheader=AUTHORIZATION: basic {basic}",
                        "-C", str(d), "push", f"https://github.com/{owner}/{name}.git", f"HEAD:refs/heads/{branch}"],
                       capture_output=True, text=True, timeout=300, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    err = re.sub(r"basic [A-Za-z0-9+/=]+", "basic ***", p.stderr)
    if p.returncode != 0:
        return f"git_push failed: {_clip(err, 600)}"
    return f"Pushed to {owner}/{name} branch '{branch}'.\n{_clip(err, 400)}\nOpen a pull request with github_pr(repo='{owner}/{name}', head='{branch}', title=...)."


def github_pr(A, args, chat_id):
    repo = (args.get("repo") or "").strip()
    if not REPO_RE.match(repo) or not REF_RE.match(args.get("head") or ""):
        return "github_pr: give repo as owner/name and a valid head branch."
    if not _allowed_owner(repo.split("/")[0]):
        return "github_pr: only repositories owned by the configured GitHub owner are allowed."
    cmd = ["gh", "pr", "create", "--repo", repo, "--head", args["head"], "--title", args["title"], "--body", args.get("body") or "Created by the cluster IDE agent."]
    if args.get("base"):
        cmd += ["--base", args["base"]]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return (p.stdout.strip() or p.stderr.strip())[:800]


def _reaper():
    """Kill sandbox processes (user llmtools) that have been running for more than 40 minutes: leftovers of background servers."""
    while True:
        time.sleep(180)
        try:
            out = subprocess.run(["ps", "-u", "llmtools", "-o", "pid=,etimes=,comm="], capture_output=True, text=True, timeout=20).stdout
            for line in out.splitlines():
                pid, et, comm = line.split(None, 2)
                if int(et) > 2400 and comm.strip() not in ("systemd", "(sd-pam)"):
                    subprocess.run(["sudo", "-n", "kill", "-9", pid], capture_output=True, timeout=10)
        except Exception:
            pass


# ---------- install ----------

def install(A):
    """Patch the running agent_server module: extra tools, extra system prompt, tool dispatch."""
    if getattr(A, "_ide_installed", False):
        return
    A._ide_installed = True
    A.TOOLS.extend(IDE_TOOLS)
    A.SYSTEM = A.SYSTEM + IDE_SYSTEM
    for t in A.TOOLS:                                   # the sandbox now allows 30-minute commands (see SANDBOX_MAX_TIMEOUT)
        if t["function"]["name"] == "run_shell":
            t["function"]["description"] = t["function"]["description"].replace("max 600", "max 1800")
            t["function"]["parameters"]["properties"]["timeout"]["description"] = "seconds, default 120, max 1800"
    threading.Thread(target=_reaper, daemon=True).start()
    orig_call_tool = A.call_tool

    def call_tool(name, args, chat_id=None, token=None):
        try:
            if name == "delegate_code":
                return delegate_code(A, args, chat_id)
            if name == "git_clone":
                return git_clone(A, args, chat_id)
            if name == "run_on_agx":
                return run_on_agx(A, args, chat_id)
            if name == "git_push":
                return git_push(A, args, chat_id)
            if name == "github_pr":
                return github_pr(A, args, chat_id)
        except Exception as e:                       # never crash the chat loop
            return f"Tool error: {e}"
        return orig_call_tool(name, args, chat_id, token)

    A.call_tool = call_tool

    orig_model_info = A.model_info

    def model_info():
        v = orig_model_info()
        if v.get("up"):
            return v
        return planner_info(A) or v

    A.model_info = model_info
