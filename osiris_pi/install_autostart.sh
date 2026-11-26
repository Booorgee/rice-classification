#!/bin/bash
# Makes the OSIRIS window open automatically when the Pi starts, and adds it
# to the applications menu. Run once on the Pi:  ./install_autostart.sh
# Undo:  ./install_autostart.sh --remove
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
PY="$HOME/rice-env/bin/python"
AUTOSTART="$HOME/.config/autostart/osiris.desktop"
MENU="$HOME/.local/share/applications/osiris.desktop"

if [ "$1" = "--remove" ]; then
    rm -f "$AUTOSTART" "$MENU"
    echo "Removed: OSIRIS no longer starts at boot."
    exit 0
fi

[ -x "$PY" ] || { echo "!! $PY not found: do Step 5 of README.md first"; exit 1; }

# Tkinter (the window toolkit) — usually preinstalled on the desktop image
if ! "$PY" -c "import tkinter" 2>/dev/null; then
    echo "Installing Tkinter..."
    sudo apt install -y python3-tk
fi

mkdir -p "$(dirname "$AUTOSTART")" "$(dirname "$MENU")"
# flock: never two windows at once; sleep: let the desktop finish loading
ENTRY="[Desktop Entry]
Type=Application
Name=OSIRIS
Comment=Rice grain inspection
Exec=sh -c 'sleep 5; cd \"$DIR\" && flock -n /tmp/osiris.lock \"$PY\" osiris_gui.py >> \"$DIR/osiris_gui.log\" 2>&1'
Terminal=false
Categories=Utility;"
echo "$ENTRY" > "$AUTOSTART"
echo "$ENTRY" | sed 's/sleep 5; //' > "$MENU"

# The window needs the desktop. The boot setting is NOT changed here: the
# raspi-config option codes differ between OS versions, and a wrong one boots
# to the text console instead of the desktop.
echo
echo "Done. OSIRIS opens by itself whenever the desktop starts."
if [ "$(systemctl get-default)" != "graphical.target" ]; then
    echo "!! This Pi does not boot to the desktop. Fix:"
    echo "     sudo systemctl set-default graphical.target"
fi
echo "For it to open without anyone logging in, choose"
echo "  sudo raspi-config  >  System Options  >  Boot / Auto Login  >  Desktop Autologin"
echo "Then restart:  sudo reboot"
echo "It is also in the applications menu (Raspberry menu > Accessories)."
echo "Problems are logged in $DIR/osiris_gui.log"
