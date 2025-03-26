# #!/usr/bin/env python3
# import os
# import sys

# # Ensure the script receives one argument
# if len(sys.argv) != 2:
#     print("Usage: compose_table.py <root_dir>")
#     sys.exit(1)

# root_dir = sys.argv[1]  # Get root directory from command-line arguments

# # Metrics to be collected
# metrics = ['psnr', 'ssim', 'lpips']

# def get_metric_from_file(filepath):
#     """Extract the last line's float value as the metric from a file."""
#     if not os.path.exists(filepath):
#         return None  # Return None if the file does not exist
#     with open(filepath, 'r') as f:
#         lines = f.readlines()
#         # Assume the last line contains the result; extract the value
#         if lines:
#             return lines[-1].strip()  # Return the last line's value
#     return None

# def collect_results(exp_path, metrics):
#     """Collect all metrics for a given experiment."""
#     result = {}
#     base_dir = os.path.join(exp_path, 'img_eval')

#     for metric in metrics:
#         result_file = os.path.join(base_dir, f"{metric}_est.txt")  # Assume file name format is f"{metric}_est.txt"
#         metric_value = get_metric_from_file(result_file)
        
#         # Format the metric value to two decimal places if it's a number
#         if metric_value is not None:
#             try:
#                 metric_value = float(metric_value)
#                 result[metric] = f"{metric_value:.2f}"  # Format to two decimal places
#             except ValueError:
#                 result[metric] = "N/A"  # Handle potential conversion errors
#         else:
#             result[metric] = "N/A"
    
#     return result

# def format_experiment_name(exp_name):
#     """Format the experiment name by taking the last segment after '-'."""
#     return exp_name.split('-')[-1]  # Get the last part after the last '-'

# def generate_latex_table(root_dir, metrics, output_file):
#     """Generate a LaTeX table containing experiment results and save to a .txt file."""
#     exp_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    

#     latex_content_head = r"""\documentclass{article}
#     \usepackage{graphicx}  % For including graphics
#     \usepackage{booktabs}  % For better table lines
#     \usepackage{caption}   % For captions in tables and figures
    
#     \begin{document}
#     """
#     with open(output_file, 'w') as f:
#         f.write(latex_content_head)
        
#     with open(output_file, 'w') as f:
#         latex_content_head = r"""\documentclass{article}
#         \usepackage{graphicx}  % For including graphics
#         \usepackage{booktabs}  % For better table lines
#         \usepackage{caption}   % For captions in tables and figures
        
#         \begin{document}
#         """
#         f.write(latex_content_head)
        
#         f.write(r'\begin{table}[h]' + '\n')
#         f.write(r'\centering' + '\n')
#         f.write(r'\begin{tabular}{|c|' + '|'.join(['c'] * len(metrics)) + '|}' + '\n')
#         f.write(r'\hline' + '\n')
        
#         # Write table header
#         f.write(' & '.join(['Experiment'] + metrics) + r'\\' + '\n')
#         f.write(r'\hline' + '\n')
        
#         # Write results for each experiment
#         for exp in exp_dirs:
#             exp_path = os.path.join(root_dir, exp)
#             results = collect_results(exp_path, metrics)
#             formatted_name = format_experiment_name(exp)  # Format experiment name
#             f.write(formatted_name)  # Experiment name
#             for metric in metrics:
#                 f.write(f" & ${results[metric]}$")
#             f.write(r'\\' + '\n')
        
#         f.write(r'\hline' + '\n')
#         f.write(r'\end{tabular}' + '\n')
#         f.write(r'\caption{Summary of metrics for experiments}' + '\n')
#         f.write(r'\label{tab:results}' + '\n')
#         f.write(r'\end{table}' + '\n')
        
#         # Close the LaTeX table
#         latex_content_tail = r"""\hline
#         \end{tabular}
#         \caption{Summary of metrics for experiments}
#         \label{tab:results}
#         \end{table}

#         \end{document}
#         """
#         f.write(latex_content_tail)
    
    

# if __name__ == "__main__":
#     output_file = os.path.join(root_dir, 'results_table.tex')
#     generate_latex_table(root_dir, metrics, output_file)
#     print(f"LaTeX table saved to {output_file}")



#!/usr/bin/env python3
import os
import sys

# Ensure the script receives one argument
if len(sys.argv) != 2:
    print("Usage: compose_table.py <root_dir>")
    sys.exit(1)

root_dir = sys.argv[1]  # Get root directory from command-line arguments

# Metrics to be collected
metrics = ['psnr', 'ssim', 'lpips']

def get_metric_from_file(filepath):
    """Extract the last line's float value as the metric from a file."""
    if not os.path.exists(filepath):
        return None  # Return None if the file does not exist
    with open(filepath, 'r') as f:
        lines = f.readlines()
        # Assume the last line contains the result; extract the value
        if lines:
            return lines[-1].strip()  # Return the last line's value
    return None

def collect_results(exp_path, metrics):
    """Collect all metrics for a given experiment."""
    result = {}
    base_dir = os.path.join(exp_path, 'img_eval')

    for metric in metrics:
        result_file = os.path.join(base_dir, f"{metric}_est.txt")  # Assume file name format is f"{metric}_est.txt"
        metric_value = get_metric_from_file(result_file)
        
        # Format the metric value to two decimal places if it's a number
        if metric_value is not None:
            try:
                metric_value = float(metric_value)
                result[metric] = f"{metric_value:.2f}"  # Format to two decimal places
            except ValueError:
                result[metric] = "N/A"  # Handle potential conversion errors
        else:
            result[metric] = "N/A"
    
    return result

def format_experiment_name(exp_name):
    """Format the experiment name by taking the last segment after '-'."""
    return exp_name.split('-')[-1]  # Get the last part after the last '-'

def generate_latex_table(root_dir, metrics, output_file):
    """Generate a LaTeX table containing experiment results and save to a .tex file."""
    exp_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    # Prepare LaTeX document header
    latex_content = r"""\documentclass{article}
\usepackage{graphicx}  % For including graphics
\usepackage{booktabs}  % For better table lines
\usepackage{caption}   % For captions in tables and figures

\begin{document}

\begin{table}[h]
\centering
\begin{tabular}{|c|""" + '|'.join(['c'] * len(metrics)) + r'|}' + '\n' + \
    r'\hline' + '\n' + \
    ' & '.join(['Experiment'] + metrics) + r'\\' + '\n' + \
    r'\hline' + '\n'

    # Write results for each experiment
    for exp in exp_dirs:
        exp_path = os.path.join(root_dir, exp)
        results = collect_results(exp_path, metrics)
        # formatted_name = format_experiment_name(exp)  # Format experiment name
        formatted_name = exp
        
        # Write the formatted name and metrics to the LaTeX table
        latex_content += formatted_name  # Experiment name
        for metric in metrics:
            latex_content += f" & ${results[metric]}$"
        latex_content += r'\\' + '\n'
    
    # Close the LaTeX table
    latex_content += r'\hline' + '\n' + r'\end{tabular}' + '\n'
    latex_content += r'\caption{Summary of metrics for experiments}' + '\n'
    latex_content += r'\label{tab:results}' + '\n'
    latex_content += r'\end{table}' + '\n'
    latex_content += r'\end{document}' + '\n'

    # Write the entire LaTeX content to the output file
    with open(output_file, 'w') as f:
        f.write(latex_content)

if __name__ == "__main__":
    output_file = os.path.join(root_dir, 'results_table.tex')
    generate_latex_table(root_dir, metrics, output_file)
    print(f"LaTeX table saved to {output_file}")

