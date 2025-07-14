import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

def plot_financial_data(ar_values, ap_values, inv_values, revenue_values, periods):
    plt.figure(figsize=(10, 6))
    plt.plot(periods, ar_values, marker='o', label='Accounts Receivable (AR)')
    plt.plot(periods, ap_values, marker='o', label='Accounts Payable (AP)')
    plt.plot(periods, inv_values, marker='o', label='Inventory (Inv)')
    plt.title('Working Capital Trends as % of Revenue')
    plt.xlabel('Period')
    plt.ylabel('% of Rev')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0f}%'.format(y)))
    plot_path = 'static/financial_data.png'
    plt.savefig(plot_path)
    plt.close()
    return plot_path
