#!/usr/bin/env python3
"""Tiny TCP forwarder (asyncio, stdlib only): listen on one or more addresses, relay every connection to one target.

Used to put a llama-server that listens on 127.0.0.1 behind several interfaces (cable, Tailscale) without exposing it
on the LAN, and to point the mini's old model port at the AGX.

usage: tcpfwd.py --listen 10.10.13.1:8080 --listen 100.94.46.75:8080 --to 127.0.0.1:8088
"""
import argparse
import asyncio


def hostport(s):
    host, port = s.rsplit(":", 1)
    return host, int(port)


async def pipe(reader, writer):
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError, OSError):
        pass
    finally:
        try:
            writer.close()
        except OSError:
            pass


async def handle(creader, cwriter, target):
    try:
        treader, twriter = await asyncio.wait_for(asyncio.open_connection(*target), timeout=10)
    except (OSError, asyncio.TimeoutError):
        cwriter.close()
        return
    await asyncio.gather(pipe(creader, twriter), pipe(treader, cwriter))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", action="append", required=True, help="host:port (repeatable)")
    ap.add_argument("--to", required=True, help="host:port")
    args = ap.parse_args()
    target = hostport(args.to)
    servers = []
    for spec in args.listen:
        host, port = hostport(spec)
        for attempt in range(60):                       # an interface (e.g. tailscale0) may not be up yet at boot
            try:
                servers.append(await asyncio.start_server(lambda r, w: handle(r, w, target), host, port))
                print(f"forwarding {host}:{port} -> {args.to}", flush=True)
                break
            except OSError as e:
                if attempt == 59:
                    print(f"cannot listen on {spec}: {e}", flush=True)
                await asyncio.sleep(2)
    if not servers:
        raise SystemExit(1)
    await asyncio.gather(*(s.serve_forever() for s in servers))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
