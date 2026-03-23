import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def analyze_latent_financial_relation():
    latent_path = "results/latent_analysis_table.csv"
    features_path = "data/FPT_features.csv"

    if not os.path.exists(latent_path) or not os.path.exists(features_path):
        print("Lỗi: Thiếu file dữ liệu cần thiết.")
        return

    # 1. Load dữ liệu
    df_latent = pd.read_csv(latent_path)
    df_features = pd.read_csv(features_path)

    df_latent['Date'] = pd.to_datetime(df_latent['Date'])
    df_features['Date'] = pd.to_datetime(df_features['Date'])

    # 2. Merge dữ liệu
    # Tài chính gốc: log_return (Returns), volatility_10 (Volatility), volume (Liquidity)
    financial_cols = ['Date', 'log_return', 'volatility_10', 'volume']
    df_merged = pd.merge(df_latent[['Date', 'mu_1', 'mu_7']], df_features[financial_cols], on='Date', how='inner')

    # 3. Tính toán tương quan
    corr_matrix = df_merged[['mu_1', 'mu_7', 'log_return', 'volatility_10', 'volume']].corr()
    
    print("\n=== MA TRẬN TƯƠNG QUAN (CORRELATION MATRIX) ===")
    print(corr_matrix[['mu_1', 'mu_7']].round(4))

    # 4. Vẽ Heatmap
    fig_heatmap = px.imshow(
        corr_matrix,
        text_auto=".3f",
        color_continuous_scale='RdBu_r',
        title="Heatmap: Tương quan giữa Biến ẩn và Chỉ số tài chính",
        labels=dict(color="Correlation")
    )

    # 5. Dashboard chi tiết: Rolling Relation & Scatter
    fig_dash = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Scatter: mu_1 vs Returns", "Scatter: mu_7 vs Returns",
            "Scatter: mu_1 vs Volatility", "Scatter: mu_7 vs Volatility",
            "Scatter: mu_1 vs Volume (Liquidity)", "Scatter: mu_7 vs Volume"
        ),
        vertical_spacing=0.1
    )

    pairs = [
        ('mu_1', 'log_return', 1, 1, '#1f77b4'), ('mu_7', 'log_return', 1, 2, '#ff7f0e'),
        ('mu_1', 'volatility_10', 2, 1, '#2ca02c'), ('mu_7', 'volatility_10', 2, 2, '#d62728'),
        ('mu_1', 'volume', 3, 1, '#9467bd'), ('mu_7', 'volume', 3, 2, '#8c564b')
    ]

    for col_x, col_y, r, c, color in pairs:
        fig_dash.add_trace(go.Scatter(
            x=df_merged[col_x], y=df_merged[col_y],
            mode='markers', marker=dict(size=3, color=color, opacity=0.4),
            name=f"{col_x} vs {col_y}"
        ), row=r, col=c)
        
        # Thêm trendline (hồi quy đơn giản)
        mask = ~df_merged[col_x].isna() & ~df_merged[col_y].isna()
        z = np.polyfit(df_merged[col_x][mask], df_merged[col_y][mask], 1)
        p = np.poly1d(z)
        x_range = np.linspace(df_merged[col_x].min(), df_merged[col_x].max(), 100)
        fig_dash.add_trace(go.Scatter(
            x=x_range, y=p(x_range),
            mode='lines', line=dict(color='black', width=1),
            showlegend=False
        ), row=r, col=c)

    fig_dash.update_layout(height=1200, title_text="<b>Phân tích chi tiết tương quan Biến ẩn & Tài chính</b>", template="plotly_white", showlegend=False)

    # 6. Lưu kết quả
    output_html = "results/latent_financial_relation.html"
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(fig_heatmap.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(fig_dash.to_html(full_html=False, include_plotlyjs=False))
    
    print(f"\nBáo cáo đã được tạo tại: {output_html}")
    print("Mời bạn mở file bằng trình duyệt để xem chi tiết tương quan.")

if __name__ == "__main__":
    analyze_latent_financial_relation()
