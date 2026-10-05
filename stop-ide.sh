#!/bin/bash
# Stop the IDE services (planner forward, Maple workers, router, login proxy). The agent keeps running (its IDE tools just report "worker unavailable").
systemctl --user stop ide-auth llm-planner llm-router llm-maple 2>/dev/null || true
ssh -o BatchMode=yes -o ConnectTimeout=5 "${NODE2:-node2@100.109.16.15}" 'systemctl --user stop maple-replica 2>/dev/null' 2>/dev/null || true
echo "IDE stopped. Bring the old models back with start-maple.sh / start-ep.sh."
