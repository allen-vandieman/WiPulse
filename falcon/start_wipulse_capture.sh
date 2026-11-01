#!/bin/bash

SESSION="wipulse_capture"

# Kill existing session if it exists
tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"

# Create session
tmux new-session -d -s "$SESSION"

# Split into top/bottom panes
tmux split-window -v -t "$SESSION"

COMMON_SETUP='cd ~/work && source env/bin/activate && cd WiPulse/echo'

# Top pane (0)
tmux send-keys -t "$SESSION:0.0" "$COMMON_SETUP" C-m
tmux send-keys -t "$SESSION:0.0" "sudo python live_monitor.py --iface wlan0 --chip 43455c0 --bw 80 --capture-mode complex" C-m

# Wait for startup
sleep 3

# Start recording in top pane
tmux send-keys -t "$SESSION:0.0" "c"

# Bottom pane (1)
tmux send-keys -t "$SESSION:0.1" "$COMMON_SETUP" C-m
tmux send-keys -t "$SESSION:0.1" "python pi_serial_recorder.py" C-m

# Wait for startup
sleep 3

# Start recording in bottom pane
tmux send-keys -t "$SESSION:0.1" "c"

# Focus bottom pane
tmux select-pane -t "$SESSION:0.1"

# Attach
tmux attach-session -t "$SESSION"