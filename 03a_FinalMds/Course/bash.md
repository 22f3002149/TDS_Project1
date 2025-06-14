## Terminal: Bash

UNIX shells are the de facto standard in the data science world and [Bash](https://www.gnu.org/software/bash/) is the most popular.
This is available by default on Mac and Linux.

On Windows, install [Git Bash](https://git-scm.com/downloads) or [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) to get a UNIX shell.

Watch this video to install WSL (12 min).

[How to Install Ubuntu on Windows 10 (WSL) (12 min): Here's a detailed description of the image:\n\n* **Composition:** The image showcases two distinct operating system logos positioned side-by-side on a gradient background.\n* **Left Logo:** On the left is the logo for Ubuntu, a Linux distribution. It's a circular design featuring three orange figures interlocked in a chain-like formation, enclosed within a white circle.\n* **Right Logo:** To the right is the Windows logo, represented by a square divided into four equally sized squares of varying shades of blue.\n* **Background:** The background is a gradient transitioning from a darker maroon on the left to a warm orange-red on the right, creating a visually dynamic backdrop.\n* **Elements between the logos:** Between the Ubuntu and Windows logos, there are small white dots that look like a dashed line, possibly suggesting a connection or transition between the two systems.\n* **Overall Impression:** The image seems to compare or relate the Ubuntu and Windows operating systems, potentially hinting at dual-booting or compatibility."](https://youtu.be_X-DHaQLrBi8)

Watch this video to understand the basics of Bash and UNIX shell commands (75 min).

[Beginner's Guide to the Bash Terminal (75 min): Here\'s a detailed description of the image:\n\n**Overall Impression:** The image is a graphic or promotional material for a beginner\'s guide to the Bash terminal. \n\n**Visual Elements:**\n\n* **Background:** The background is a solid, deep purple color.\n* **Icon:** A hexagonal icon is displayed on the left side. The hexagon is split into two halves. One half is white, the other is dark green with a white dollar sign ($) symbol within it.\n* **Text:** To the right of the icon, white text is prominently displayed, reading:\n * "Beginner\'s" (on the first line)\n * "Guide to" (on the second line)\n * "the Bash" (on the third line)\n * "Terminal" (on the fourth line)\n\n**Style & Tone:** The image has a clean and modern look. The design suggests it’s aimed at someone new to the Bash terminal, providing introductory guidance.\n\n\n\n](https://youtu.be/oxuRxtrO2Ag)

Essential Commands:

```bash
# File Operations
ls -la               # List all files with details
cd path/to/dir       # Change directory
pwd                  # Print working directory
cp source dest       # Copy files
mv source dest       # Move/rename files
rm -rf dir           # Remove directory recursively

# Text Processing
grep "pattern" file  # Search for pattern
sed 's/old/new/' f   # Replace text
awk '{print $1}' f   # Process text by columns
cat file | wc -l     # Count lines

# Process Management
ps aux               # List processes
kill -9 PID          # Force kill process
top                  # Monitor processes
htop                 # Interactive process viewer

# Network
curl url             # HTTP requests
wget url             # Download files
nc -zv host port     # Test connectivity
ssh user@host        # Remote login

# Count unique values in CSV column
cut -d',' -f1 data.csv | sort | uniq -c

# Quick data analysis
awk -F',' '{sum+=$2} END {print sum/NR}' data.csv  # Average
sort -t',' -k2 -n data.csv | head                  # Top 10

# Monitor log in real-time
tail -f log.txt | grep --color 'ERROR'
```

Bash Scripting Essentials:

```bash
#!/bin/bash

# Variables
NAME="value"
echo $NAME

# Loops
for i in {1..5}; do
    echo $i
done

# Conditionals
if [ -f "file.txt" ]; then
    echo "File exists"
fi

# Functions
process_data() {
    local input=$1
    echo "Processing $input"
}
```

Productivity Tips:

1. **Command History**

   ```bash
   history         # Show command history
   Ctrl+R         # Search history
   !!             # Repeat last command
   !$             # Last argument
   ```

2. **Directory Navigation**

   ```bash
   pushd dir      # Push directory to stack
   popd           # Pop directory from stack
   cd -           # Go to previous directory
   ```

3. **Job Control**

   ```bash
   command &      # Run in background
   Ctrl+Z         # Suspend process
   bg             # Resume in background
   fg             # Resume in foreground
   ```

4. **Useful Aliases** - typically added to `~/.bashrc`
   ```bash
   alias ll='ls -la'
   alias gs='git status'
   alias jupyter='jupyter notebook'
   alias activate='source venv/bin/activate'
   ```
