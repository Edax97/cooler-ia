#!/bin/bash
sudo chmod +x run_app.sh
sudo cp run_app.sh /usr/local/bin/
sudo cp run_cooler.service cooler.timer "$HOME/.config/systemd/user/"
systemctl --user daemon-reload
systemctl --user enable cooler.timer
systemctl --user restart cooler.timer