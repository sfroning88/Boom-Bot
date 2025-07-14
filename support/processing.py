import math
import re
import pandas as pd
import numpy as np

def summarize_values(values):
    if not values:
        return "no data"
    if values and isinstance(values[0], list):
        flat = []
        for sublist in values:
            for item in sublist:
                s = str(item).strip()
                if s in ('', '-', 'nan', 'NaN', 'None'):
                    continue
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

def calculate_working_capital_ratios(ar_total, ap_total, inv_total, revenue_total):
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

def process_file(file):
    def extract_periods_from_headers(df, max_scan_rows=10):
        periods = []
        period_col_indices = []
        header_row_idx = -1
        date_pattern = re.compile(r'((0?[1-9]|1[0-2])[/-](0?[1-9]|[12]\d|3[01])[/-]?\d{2,4}|(20\d{2})\s*Q[1-4]|Q[1-4]\s*(20\d{2})|\b(20\d{2})\b)')
        for row_idx in range(min(max_scan_rows, len(df))):
            row = df.iloc[row_idx]
            found_dates = []
            found_indices = []
            for col_idx, cell_value in enumerate(row):
                cell_str = str(cell_value).strip().lower()
                if not cell_str or cell_str in ['nan', 'none', '']:
                    continue
                # Special case: if 'debit' or 'credit' in cell, treat as a period but mark for reconciliation
                if 'debit' in cell_str or 'credit' in cell_str:
                    # Remove 'debit'/'credit' for period label
                    period_label = re.sub(r'\s*(debit|credit)\s*', '', cell_str, flags=re.IGNORECASE).strip()
                    found_dates.append(period_label)
                    found_indices.append(col_idx)
                elif date_pattern.search(cell_str):
                    found_dates.append(cell_str)
                    found_indices.append(col_idx)
                elif is_date_cell(cell_str):
                    period_label = extract_period_label(cell_str)
                    if period_label:
                        found_dates.append(period_label)
                        found_indices.append(col_idx)
            if len(found_dates) >= 2:
                seen = set()
                periods = [x for x in found_dates if not (x in seen or seen.add(x))]
                period_col_indices = [col_idx for i, col_idx in enumerate(found_indices) if found_dates[i] in periods]
                header_row_idx = row_idx
                break
        if not periods and len(df.columns) > 1:
            for col_idx, col in enumerate(df.columns):
                col_str = str(col).strip().lower()
                if 'debit' in col_str or 'credit' in col_str:
                    period_label = re.sub(r'\s*(debit|credit)\s*', '', col_str, flags=re.IGNORECASE).strip()
                    periods.append(period_label)
                    period_col_indices.append(col_idx)
                elif date_pattern.search(col_str):
                    periods.append(col_str)
                    period_col_indices.append(col_idx)
                elif is_date_cell(col_str):
                    period_label = extract_period_label(col_str)
                    if period_label and period_label not in periods:
                        periods.append(period_label)
                        period_col_indices.append(col_idx)
        if not periods:
            num_cols = len(df.columns)
            estimated_periods = min(5, num_cols)
            periods = [f"Period {i+1}" for i in range(estimated_periods)]
            period_col_indices = list(range(2, 2+estimated_periods))
            header_row_idx = 0
        return periods, period_col_indices, header_row_idx + 1
    def is_date_cell(cell_str):
        if re.search(r'^\d+\.\d+$', cell_str):
            return False
        date_patterns = [
            r'\b(20\d{2})\b', r'\bFY\s*(20\d{2})\b',
            r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*(20\d{2})\b',
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s*(20\d{2})\b',
            r'\bQ[1-4]\s*(20\d{2})\b', r'\bQ[1-4]\s*FY\s*(20\d{2})\b',
            r'\b(20\d{2})[-/](20\d{2})\b', r'\b(20\d{2})[/-](0?[1-9]|1[0-2])\b',
            r'\b(0?[1-9]|1[0-2])[/-](20\d{2})\b',
            r'\b(0?[1-9]|[12]\d|3[01])[/-](0?[1-9]|1[0-2])[/-](20\d{2})\b',
            r'\b(0?[1-9]|[12]\d|3[01])[/-](0?[1-9]|1[0-2])[/-](20\d{2})\b',
        ]
        for pattern in date_patterns:
            if re.search(pattern, cell_str, re.IGNORECASE):
                return True
        return False
    def extract_period_label(cell_str):
        year_match = re.search(r'\b(20\d{2})\b', cell_str)
        if year_match:
            year = year_match.group(1)
            month_patterns = {
                r'\bJan[a-z]*\b': 'Jan', r'\bFeb[a-z]*\b': 'Feb', r'\bMar[a-z]*\b': 'Mar',
                r'\bApr[a-z]*\b': 'Apr', r'\bMay[a-z]*\b': 'May', r'\bJun[a-z]*\b': 'Jun',
                r'\bJul[a-z]*\b': 'Jul', r'\bAug[a-z]*\b': 'Aug', r'\bSep[a-z]*\b': 'Sep',
                r'\bOct[a-z]*\b': 'Oct', r'\bNov[a-z]*\b': 'Nov', r'\bDec[a-z]*\b': 'Dec'
            }
            for pattern, month in month_patterns.items():
                if re.search(pattern, cell_str, re.IGNORECASE):
                    return f"{month} {year}"
            quarter_match = re.search(r'\bQ([1-4])\b', cell_str, re.IGNORECASE)
            if quarter_match:
                quarter = quarter_match.group(1)
                return f"Q{quarter} {year}"
            if re.search(r'\bFY\b', cell_str, re.IGNORECASE):
                return f"FY {year}"
            return year
        if re.search(r'\b(0?[1-9]|1[0-2])[/-](0?[1-9]|[12]\d|3[01])[/-](20\d{2})\b', cell_str):
            match = re.search(r'\b(0?[1-9]|1[0-2])[/-](0?[1-9]|[12]\d|3[01])[/-](20\d{2})\b', cell_str)
            if match:
                month, day, year = match.groups()
                return f"{month}/{year}"
        return cell_str.strip()
    if file.filename.lower().endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    periods, period_col_indices, header_rows = extract_periods_from_headers(df)
    # Reconcile periods if any contain 'debit' or 'credit' (merge into one period per date)
    period_map = {}
    for i, period in enumerate(periods):
        base_period = re.sub(r'\s*(debit|credit)\s*', '', period, flags=re.IGNORECASE).strip()
        if base_period not in period_map:
            period_map[base_period] = []
        period_map[base_period].append(i)
    merged_periods = list(period_map.keys())
    merged_period_col_indices = [period_col_indices[indices[0]] for indices in period_map.values()]
    periods = merged_periods
    period_col_indices = merged_period_col_indices
    data_df = df.iloc[header_rows:].reset_index(drop=True)
    nwc = {period: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for period in periods}
    period_to_col = {period: col_idx for period, col_idx in zip(periods, period_col_indices)}
    from static.definitions import ar_patterns, ap_patterns, inv_patterns, rev_patterns, rev_exclude
    for idx, row in data_df.iterrows():
        row_str = ' '.join([str(x).lower() for x in row.astype(str)])
        row_str_nopunct = re.sub(r'[^a-z0-9 ]', ' ', row_str)
        account_type = str(row.iloc[1]).strip().upper() if len(row) > 1 else ''
        account_name = str(row.iloc[0]).strip().lower() if len(row) > 0 else ''
        for period, col_idx in period_to_col.items():
            # If this period was merged from debit/credit, check both columns
            indices = period_map[period] if period in period_map else [col_idx]
            value = None
            for i in indices:
                if i < len(row):
                    try:
                        v = float(row.iloc[i])
                        if math.isfinite(v) and (value is None or abs(v) > abs(value)):
                            value = v
                    except (ValueError, TypeError):
                        continue
            if account_type == 'IS' and any(re.search(p, account_name) for p in rev_patterns) and not any(re.search(p, account_name) for p in rev_exclude):
                if value is not None:
                    nwc[period][3] += value
            elif account_type == 'BS':
                if any(re.search(p, row_str_nopunct) for p in ar_patterns):
                    if value is not None:
                        nwc[period][0] += value
                if any(re.search(p, row_str_nopunct) for p in ap_patterns):
                    if value is not None:
                        nwc[period][1] += value
                if any(re.search(p, row_str_nopunct) for p in inv_patterns):
                    if value is not None:
                        nwc[period][2] += value
    for period in periods:
        ar, ap, inv, rev = nwc[period][:4]
        ar_ratio = (ar / rev * 100) if rev else 0
        ap_ratio = (ap / rev * 100) if rev else 0
        inv_ratio = (inv / rev * 100) if rev else 0
        working_capital_cycle = ar_ratio + inv_ratio - ap_ratio
        nwc[period][4] = ar_ratio
        nwc[period][5] = ap_ratio
        nwc[period][6] = inv_ratio
        nwc[period][7] = working_capital_cycle
    return periods, nwc
