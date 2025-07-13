import os
import math
import re
from datetime import datetime

from flask import Flask, render_template, request, jsonify

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pipelines import pipeline
import torch

from werkzeug.utils import secure_filename
import tempfile
import pandas as pd

import matplotlib
matplotlib.use('Agg')

ALLOWED_EXTENSIONS = {'csv', 'xls', 'xlsx'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# load in hugging face token
hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
if hf_token is None:
    raise ValueError("Missing Hugging Face token. Set HUGGINGFACEHUB_API_TOKEN in your environment.")

# free model good with numerical analysis
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# initialize the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# create a text generation pipeline
pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer,
    device=-1 # CPU
)

# create a Flask app
app = Flask(__name__)

# set the template folder with chat formatting
@app.route("/")
def index():
    return render_template('chat.html')

# handle chat requests
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)

# function to get chat response
def get_Chat_response(prompt):
    formatted_prompt = f"<|system|>You are a helpful financial analyst assistant.</s><|user|>{prompt}</s><|assistant|>"
    output = pipe(formatted_prompt, max_new_tokens=1024, do_sample=True, temperature=0.7)
    return output[0]['generated_text'].split('<|assistant|>')[-1].strip()

# function to summarize values
def summarize_values(values):
    import numpy as np
    if not values:
        return "no data"
    
    # Handle both nested lists and simple lists
    if values and isinstance(values[0], list):
        # Old format: nested lists
        flat = []
        for sublist in values:
            for item in sublist:
                # Convert to string for parsing
                s = str(item).strip()
                if s in ('', '-', 'nan', 'NaN', 'None'):
                    continue
                # Handle parentheses as negatives
                if s.startswith('(') and s.endswith(')'):
                    try:
                        num = -float(s[1:-1].replace(',', ''))
                    except Exception:
                        continue
                else:
                    try:
                        num = float(s.replace(',', ''))
                    except Exception:
                        continue
                if math.isfinite(num):
                    flat.append(num)
    else:
        # New format: simple list of numbers
        flat = []
        for item in values:
            if isinstance(item, (int, float)) and math.isfinite(item):
                flat.append(item)
    
    if not flat:
        return "no data"
    def usd(x):
        return f"${x:,.2f}"
    return (
        f"min={usd(round(np.min(flat), 2))}, "
        f"max={usd(round(np.max(flat), 2))}, "
        f"mean={usd(round(np.mean(flat), 2))}"
    )

# function to extract analysis from model output
def extract_analysis(model_output, prompt):
    if model_output.startswith(prompt):
        model_output = model_output[len(prompt):].lstrip()
    # Return more of the response and preserve formatting
    return model_output.strip()

# function to calculate working capital ratios and trends
def calculate_working_capital_ratios(ar_total, ap_total, inv_total, revenue_total):
    """Calculate working capital ratios and provide trend analysis"""
    if revenue_total <= 0:
        return {
            'ar_ratio': 0,
            'ap_ratio': 0,
            'inv_ratio': 0,
            'working_capital_cycle': 0,
            'analysis': "No revenue data available for ratio calculations."
        }
    
    ar_ratio = (ar_total / revenue_total) * 100
    ap_ratio = (ap_total / revenue_total) * 100
    inv_ratio = (inv_total / revenue_total) * 100
    
    # Basic working capital efficiency indicators
    working_capital_cycle = ar_ratio + inv_ratio - ap_ratio
    
    analysis = []
    if ar_ratio > 20:
        analysis.append(f"High AR ratio ({ar_ratio:.1f}%) suggests potential collection issues.")
    if ap_ratio > 15:
        analysis.append(f"High AP ratio ({ap_ratio:.1f}%) may indicate cash flow pressure.")
    if inv_ratio > 25:
        analysis.append(f"High inventory ratio ({inv_ratio:.1f}%) suggests potential overstocking.")
    if working_capital_cycle > 30:
        analysis.append(f"Long working capital cycle ({working_capital_cycle:.1f}%) indicates cash tied up in operations.")
    
    if not analysis:
        analysis.append("Working capital ratios appear healthy.")
    
    return {
        'ar_ratio': ar_ratio,
        'ap_ratio': ap_ratio,
        'inv_ratio': inv_ratio,
        'working_capital_cycle': working_capital_cycle,
        'analysis': ' '.join(analysis)
    }

def extract_periods_from_headers(df, max_scan_rows=10):
    """
    Extract period headers by collecting all date-like cells in the first header row with dates.
    Removes duplicates and preserves order.
    """
    periods = []
    header_row_idx = -1
    
    # Look for period headers in the first few rows only
    for row_idx in range(min(max_scan_rows, len(df))):
        row = df.iloc[row_idx]
        found_dates = []
        for col_idx, cell_value in enumerate(row):
            cell_str = str(cell_value).strip()
            if not cell_str or cell_str.lower() in ['nan', 'none', '']:
                continue
            if is_date_cell(cell_str):
                period_label = extract_period_label(cell_str)
                if period_label:
                    found_dates.append(period_label)
        if len(found_dates) >= 2:
            # Remove duplicates while preserving order
            seen = set()
            periods = [x for x in found_dates if not (x in seen or seen.add(x))]
            header_row_idx = row_idx
            break
    
    # If no timeline found, try to infer from column headers
    if not periods and len(df.columns) > 1:
        for col in df.columns:
            col_str = str(col).strip()
            if is_date_cell(col_str):
                period_label = extract_period_label(col_str)
                if period_label and period_label not in periods:
                    periods.append(period_label)
    
    # If still no periods found, create generic periods based on columns
    if not periods:
        num_cols = len(df.columns)
        estimated_periods = min(5, num_cols)  # Conservative estimate
        periods = [f"Period {i+1}" for i in range(estimated_periods)]
        header_row_idx = 0
    
    # Limit periods to a reasonable number
    if len(periods) > 24:
        periods = periods[:24]
    
    return periods, header_row_idx + 1

def is_date_cell(cell_str):
    """Check if a cell contains a date in various formats."""
    # Skip cells that look like data (decimal numbers)
    if re.search(r'^\d+\.\d+$', cell_str):
        return False
        
    date_patterns = [
        r'\b(20\d{2})\b',  # Years like 2023, 2024
        r'\bFY\s*(20\d{2})\b',  # FY 2023
        r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*(20\d{2})\b',  # Jan 2023
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s*(20\d{2})\b',  # January 2023
        r'\bQ[1-4]\s*(20\d{2})\b',  # Q1 2023
        r'\bQ[1-4]\s*FY\s*(20\d{2})\b',  # Q1 FY 2023
        r'\b(20\d{2})[-/](20\d{2})\b',  # 2023-2024
        r'\b(20\d{2})[/-](0?[1-9]|1[0-2])\b',  # 2023/01 or 2023-01
        r'\b(0?[1-9]|1[0-2])[/-](20\d{2})\b',  # 01/2023 or 1/2023
        r'\b(0?[1-9]|[12]\d|3[01])[/-](0?[1-9]|1[0-2])[/-](20\d{2})\b',  # MM/DD/YYYY
        r'\b(0?[1-9]|[12]\d|3[01])[/-](0?[1-9]|1[0-2])[/-](20\d{2})\b',  # DD/MM/YYYY
    ]
    
    for pattern in date_patterns:
        if re.search(pattern, cell_str, re.IGNORECASE):
            return True
    return False

def extract_timeline_from_row(row, start_col_idx):
    """
    Extract a continuous timeline of dates starting from start_col_idx.
    Returns a list of period labels.
    """
    timeline = []
    
    # Look forward from the start column
    for col_idx in range(start_col_idx, len(row)):
        cell_value = row.iloc[col_idx]
        cell_str = str(cell_value).strip()
        
        # Skip empty cells
        if not cell_str or cell_str.lower() in ['nan', 'none', '']:
            continue
            
        # Check if this cell contains a date
        if is_date_cell(cell_str):
            # Extract and clean the date
            period_label = extract_period_label(cell_str)
            if period_label:
                timeline.append(period_label)
        else:
            # If we hit a non-date cell, stop the timeline
            break
    
    return timeline

def extract_period_label(cell_str):
    """
    Extract a clean period label from a date cell.
    """
    # Try to extract year first
    year_match = re.search(r'\b(20\d{2})\b', cell_str)
    if year_match:
        year = year_match.group(1)
        
        # Check for month patterns
        month_patterns = {
            r'\bJan[a-z]*\b': 'Jan', r'\bFeb[a-z]*\b': 'Feb', r'\bMar[a-z]*\b': 'Mar',
            r'\bApr[a-z]*\b': 'Apr', r'\bMay[a-z]*\b': 'May', r'\bJun[a-z]*\b': 'Jun',
            r'\bJul[a-z]*\b': 'Jul', r'\bAug[a-z]*\b': 'Aug', r'\bSep[a-z]*\b': 'Sep',
            r'\bOct[a-z]*\b': 'Oct', r'\bNov[a-z]*\b': 'Nov', r'\bDec[a-z]*\b': 'Dec'
        }
        
        for pattern, month in month_patterns.items():
            if re.search(pattern, cell_str, re.IGNORECASE):
                return f"{month} {year}"
        
        # Check for quarter patterns
        quarter_match = re.search(r'\bQ([1-4])\b', cell_str, re.IGNORECASE)
        if quarter_match:
            quarter = quarter_match.group(1)
            return f"Q{quarter} {year}"
        
        # Check for FY pattern
        if re.search(r'\bFY\b', cell_str, re.IGNORECASE):
            return f"FY {year}"
        
        # Just return the year
        return year
    
    # If no year found, try to extract any meaningful date pattern
    if re.search(r'\b(0?[1-9]|1[0-2])[/-](0?[1-9]|[12]\d|3[01])[/-](20\d{2})\b', cell_str):
        # MM/DD/YYYY format
        match = re.search(r'\b(0?[1-9]|1[0-2])[/-](0?[1-9]|[12]\d|3[01])[/-](20\d{2})\b', cell_str)
        if match:
            month, day, year = match.groups()
            return f"{month}/{year}"
    
    return cell_str.strip()  # Return the original string if no pattern matches

def plot_financial_data(ap_values, ar_values, inv_values, revenue_values, periods=None, save_path="static/financial_plot.png"):
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Ensure we have matching lengths
    n = max(len(ap_values), len(ar_values), len(inv_values), len(revenue_values))
    if not periods:
        periods = [f"Period {i+1}" for i in range(n)]
    
    # Truncate periods if we have more periods than data points
    if len(periods) > n:
        periods = periods[:n]
    # Extend periods if we have more data points than periods
    elif len(periods) < n:
        periods.extend([f"Period {i+1}" for i in range(len(periods), n)])
    
    # Pad arrays to match length
    def pad(lst):
        return lst + [0] * (n - len(lst))
    ap, ar, inv, rev = map(pad, [ap_values, ar_values, inv_values, revenue_values])
    
    # Create numeric x-axis positions
    x_positions = np.arange(len(periods))
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_positions, ap, marker='o', label='Accounts Payable')
    plt.plot(x_positions, ar, marker='o', label='Accounts Receivable')
    plt.plot(x_positions, inv, marker='o', label='Inventory')
    plt.plot(x_positions, rev, marker='o', label='Revenue')
    plt.xlabel('Period')
    plt.ylabel('Amount')
    plt.title('Working Capital Line Items Over Time')
    plt.legend()
    plt.tight_layout()
    # Set x-axis ticks to show period labels
    plt.xticks(x_positions, periods, rotation=45)
    plt.savefig(save_path)
    plt.close()
    return save_path

# function to load file in chat progress
@app.route('/chat_upload', methods=['POST'])
def chat_upload():
    msg = request.form.get('msg', '')
    file = request.files.get('file')
    ap_rows, ar_rows, inv_rows, revenue_rows = [], [], [], []
    ap_values, ar_values, inv_values, revenue_values = [], [], [], []
    if file:
        filename = file.filename if file.filename is not None else ''
        if not allowed_file(filename):
            return jsonify({'success': False, 'error': 'Invalid file type. Please upload CSV or Excel files only.'}), 400
        safe_filename = secure_filename(filename)
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, safe_filename)
        file.save(file_path)
        if filename.lower().endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        # Extract periods and determine header rows
        periods, header_rows = extract_periods_from_headers(df)
        
        # Skip header rows when processing data
        data_df = df.iloc[header_rows:].reset_index(drop=True)
        
        # Limit data processing to match the number of periods
        # Most financial statements have data rows corresponding to the periods
        max_data_rows = len(periods) * 2  # Allow some flexibility for multiple line items per period
        data_df = data_df.head(max_data_rows)
        
        # Initialize arrays to store one value per period
        ap_values = [0] * len(periods)
        ar_values = [0] * len(periods)
        inv_values = [0] * len(periods)
        revenue_values = [0] * len(periods)
        
        for idx, row in data_df.iterrows():
            row_str = ' '.join([str(x).lower() for x in row.astype(str)])
            row_str_nopunct = re.sub(r'[^a-z0-9 ]', ' ', row_str)
            
            # Determine which financial category this row represents
            category = None
            if (
                re.search(r'accounts receivable', row_str_nopunct) or
                re.search(r'\ba r\b', row_str_nopunct) or
                re.search(r'\bar\b', row_str_nopunct) or
                re.search(r'\ba r\b', row_str) or
                re.search(r'\ba r\b', row_str.replace('/', '')) or
                re.search(r'receivable', row_str_nopunct)
            ):
                category = 'ar'
            elif (
                re.search(r'accounts payable', row_str_nopunct) or
                re.search(r'\ba p\b', row_str_nopunct) or
                re.search(r'\bap\b', row_str_nopunct) or
                re.search(r'\ba p\b', row_str) or
                re.search(r'\ba p\b', row_str.replace('/', '')) or
                re.search(r'payable', row_str_nopunct)
            ):
                category = 'ap'
            elif (
                re.search(r'inventory', row_str_nopunct) or
                re.search(r'stock', row_str_nopunct) or
                re.search(r'raw materials', row_str_nopunct) or
                re.search(r'finished goods', row_str_nopunct)
            ):
                category = 'inv'
            elif (
                re.search(r'revenue', row_str_nopunct) or
                re.search(r'sales', row_str_nopunct) or
                re.search(r'income', row_str_nopunct) or
                re.search(r'gross revenue', row_str_nopunct) or
                re.search(r'net revenue', row_str_nopunct) or
                re.search(r'total revenue', row_str_nopunct) or
                re.search(r'turnover', row_str_nopunct)
            ):
                category = 'rev'
            
            # If we found a category, extract the values for each period
            if category:
                # Get numeric values from the row (skip the first few columns which are usually labels)
                numeric_values = []
                for col_idx in range(2, min(len(row), len(periods) + 2)):  # Start from col 2, limit to number of periods
                    try:
                        val = float(row.iloc[col_idx])
                        if math.isfinite(val):
                            numeric_values.append(val)
                        else:
                            numeric_values.append(0)
                    except (ValueError, TypeError):
                        numeric_values.append(0)
                
                # Pad or truncate to match number of periods
                while len(numeric_values) < len(periods):
                    numeric_values.append(0)
                numeric_values = numeric_values[:len(periods)]
                
                # Add to the appropriate category
                if category == 'ar':
                    ar_values = [ar_values[i] + numeric_values[i] for i in range(len(periods))]
                elif category == 'ap':
                    ap_values = [ap_values[i] + numeric_values[i] for i in range(len(periods))]
                elif category == 'inv':
                    inv_values = [inv_values[i] + numeric_values[i] for i in range(len(periods))]
                elif category == 'rev':
                    revenue_values = [revenue_values[i] + numeric_values[i] for i in range(len(periods))]
        
        # Calculate totals for ratios
        revenue_total = sum(revenue_values) if revenue_values else 0
        ap_total = sum(ap_values) if ap_values else 0
        ar_total = sum(ar_values) if ar_values else 0
        inv_total = sum(inv_values) if inv_values else 0
        ratios = calculate_working_capital_ratios(ar_total, ap_total, inv_total, revenue_total)
        prompt = (
            "You are a financial analyst talking directly to a small business owner.\n"
            f"Time periods: {', '.join(periods)}\n"
            f"Revenue: {summarize_values(revenue_values)}\n"
            f"AP: {summarize_values(ap_values)} ({ratios['ap_ratio']:.1f}% of revenue)\n"
            f"AR: {summarize_values(ar_values)} ({ratios['ar_ratio']:.1f}% of revenue)\n"
            f"Inventory: {summarize_values(inv_values)} ({ratios['inv_ratio']:.1f}% of revenue)\n"
            f"Working Capital Cycle: {ratios['working_capital_cycle']:.1f}%\n"
            "Write a 3 sentence analysis of the working capital trends including recommendations for improvement."
        )
        model_output = get_Chat_response(prompt)
        analysis = extract_analysis(model_output, prompt)
        # Generate and save the plot with extracted periods
        plot_path = plot_financial_data(ap_values, ar_values, inv_values, revenue_values, periods)
        return jsonify({'success': True, 'analysis': analysis, 'plot_path': plot_path}), 200
    return get_Chat_response(msg)

@app.route('/chat_followup', methods=['POST'])
def chat_followup():
    msg = request.form.get('msg', '')
    context = request.form.get('context', '')
    # Compose a prompt that includes the previous analysis/context and the new user question
    prompt = (
        f"Previous analysis: {context}\n"
        f"User follow-up question: {msg}\n"
        "As a financial analyst, answer the user's follow-up question in a concise, practical way, referencing the previous analysis if relevant."
    )
    ai_response = get_Chat_response(prompt)
    return jsonify({'success': True, 'response': ai_response}), 200

if __name__ == '__main__':
    app.run()
