# write python script that checks all files in 'DisPosts' directory & remoeves multiple linebreaks to single linebreaks.
import os
import re
def remove_multiple_linebreaks(posts_dir):
    summary = {
        'total_files': 0,
        'linebreaks_removed': 0
    }

    # Iterate through all files in the posts directory
    for root, _, files in os.walk(posts_dir):
        for file in files:
            if file.endswith('.txt'):
                summary['total_files'] += 1
                file_path = os.path.join(root, file)

                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Replace multiple linebreaks with a single linebreak
                new_content, num_replacements = re.subn(r'\n{2,}', '\n', content)
                summary['linebreaks_removed'] += num_replacements

                # Write the modified content back to the file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)

    return summary
# Example usage
if __name__ == "__main__":
    posts_directory = 'DisPosts'  # Replace with your actual directory path
    summary = remove_multiple_linebreaks(posts_directory)

    # Print the summary
    print(f"Total files processed: {summary['total_files']}")
    print(f"Total linebreaks removed: {summary['linebreaks_removed']}")
    # Write the summary to a text file
    summary_file_path = 'linebreaks_summary.txt'
    with open(summary_file_path, 'w', encoding='utf-8') as summary_file:
        summary_file.write(f"Total files processed: {summary['total_files']}\n")
        summary_file.write(f"Total linebreaks removed: {summary['linebreaks_removed']}\n")
        summary_file.write("Processed files:\n")
        for root, _, files in os.walk(posts_directory):
            for file in files:
                if file.endswith('.txt'):
                    summary_file.write(f"{os.path.join(root, file)}\n")
        print(f"Summary written to {summary_file_path}")
# End of code
# This script removes multiple linebreaks from text files in the specified directory.