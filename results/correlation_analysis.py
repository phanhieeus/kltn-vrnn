import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import os

def analyze_correlation(file_path="results/latent_analysis_table.csv"):
    if not os.path.exists(file_path):
        print(f"Lỗi: Không tìm thấy file {file_path}")
        return

    # 1. Load dữ liệu
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    m1 = df['mu_1']
    m7 = df['mu_7']

    # 2. Tính toán chỉ số thống kê
    pearson_corr, _ = stats.pearsonr(m1, m7)
    spearman_corr, _ = stats.spearmanr(m1, m7)
    
    # 3. Tính Rolling Correlation (30 ngày) để xem biến thiên theo thời gian
    window = 30
    rolling_corr = m1.rolling(window=window).corr(m7)

    print("\n=== KẾT QUẢ PHÂN TÍCH TƯƠNG QUAN MU_1 VÀ MU_7 ===")
    print(f"Hệ số Pearson:  {pearson_corr:.4f} (Tương quan tuyến tính)")
    print(f"Hệ số Spearman: {spearman_corr:.4f} (Tương quan thứ hạng/phi tuyến)")
    print(f"Giai đoạn: {df['Date'].min().date()} đến {df['Date'].max().date()}")

    # 4. Tạo Dashboard trực quan
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.1,
        subplot_titles=(
            "1. Phân phối tương quan (Scatter Plot & Regression)", 
            "2. Biến động Mu_1 và Mu_7 theo thời gian",
            f"3. Tương quan trượt {window} phiên (Rolling Correlation)"
        ),
        row_heights=[0.4, 0.3, 0.3]
    )

    # Subplot 1: Scatter Plot
    fig.add_trace(go.Scatter(
        x=m1, y=m7, mode='markers',
        marker=dict(color='rgba(142, 68, 173, 0.5)', size=4),
        name='Data Points',
        hovertext=df['Date'].dt.strftime('%d/%m/%Y')
    ), row=1, col=1)

    # Thêm đường hồi quy (Trendline)
    z = np.polyfit(m1, m7, 1)
    p = np.poly1d(z)
    m1_range = np.linspace(m1.min(), m1.max(), 100)
    fig.add_trace(go.Scatter(
        x=m1_range, y=p(m1_range),
        mode='lines', line=dict(color='red', width=2),
        name='Dòng xu hướng (Linear Trend)'
    ), row=1, col=1)

    # Subplot 2: Time Series
    fig.add_trace(go.Scatter(x=df['Date'], y=m1, name='Mu_1', line=dict(color='#8e44ad')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=m7, name='Mu_7', line=dict(color='#16a085')), row=2, col=1)

    # Subplot 3: Rolling Correlation
    fig.add_trace(go.Scatter(
        x=df['Date'], y=rolling_corr,
        fill='tozeroy', name=f'Corr ({window}d)',
        line=dict(color='#e67e22')
    ), row=3, col=1)
    
    # Đường trung tâm 0 để dễ quan sát đảo chiều tương quan
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=3, col=1)

    # Cấu hình Layout
    fig.update_layout(
        height=1000,
        title_text=f"<b>Phân tích Tương quan Latent Mu_1 & Mu_7</b><br>Pearson: {pearson_corr:.3f}",
        template="plotly_white",
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Mu_1", row=1, col=1)
    fig.update_yaxes(title_text="Mu_7", row=1, col=1)
    fig.update_yaxes(title_text="Giá trị", row=2, col=1)
    fig.update_yaxes(title_text="Hệ số tương quan", range=[-1.1, 1.1], row=3, col=1)

    output_path = "results/mu1_mu7_correlation.html"
    config = {'scrollZoom': True, 'displaylogo': False}
    fig.write_html(output_path, config=config)
    print(f"\nĐã tạo báo cáo tương quan tại: {output_path}")

if __name__ == "__main__":
    analyze_correlation()
