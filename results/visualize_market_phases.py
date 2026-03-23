import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# 1. Đọc và làm sạch dữ liệu
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    
    # Chuẩn hóa cột Ngày
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    
    # Chuyển đổi Giá sang số (Xử lý dấu phẩy)
    for col in ['Price', 'Open', 'High', 'Low']:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(str).str.replace(',', '', regex=False).astype(float)
    
    # Sắp xếp theo ngày
    df = df.sort_values('Date').reset_index(drop=True)
    return df

# 2. Xác định các giai đoạn Bull/Bear (Khung thời gian ngắn)
def detect_phases(df, fast=5, slow=12):
    """
    Sử dụng EMA (Exponential Moving Average) để phản ứng nhanh hơn với biến động giá.
    Khung thời gian 5/12 phù hợp để bắt các nhịp sóng ngắn (4-7 ngày).
    """
    df['EMA_Fast'] = df['Price'].ewm(span=fast, adjust=False).mean()
    df['EMA_Slow'] = df['Price'].ewm(span=slow, adjust=False).mean()
    
    # Phase: 1 for Bull, -1 for Bear
    df['Phase'] = np.where(df['EMA_Fast'] > df['EMA_Slow'], 1, -1)
    df.loc[df['EMA_Slow'].isna(), 'Phase'] = 0
    return df

# 3. Vẽ biểu đồ tương tác
def plot_market_phases(df):
    # Tạo figure với 2 trục Y
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. Đường giá (Trục Y chính)
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Price'],
        mode='lines', line=dict(color='#2c3e50', width=2),
        name='Giá FPT (VNĐ)',
        hovertemplate='<b>Ngày:</b> %{x|%d/%m/%Y}<br><b>Giá:</b> %{y:,.0f} VNĐ<extra></extra>'
    ), secondary_y=False)

    # 2. Đường EMA (Trục Y chính)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_Fast'], line=dict(color='orange', width=1, dash='dot'), name='EMA(5)', hoverinfo='skip'), secondary_y=False)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_Slow'], line=dict(color='red', width=1, dash='dot'), name='EMA(12)', hoverinfo='skip'), secondary_y=False)

    # 3. Các cột Mu (Trục Y phụ - Latent Dynamics)
    if 'mu_1' in df.columns and 'mu_7' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['mu_1'],
            mode='lines', line=dict(color='#8e44ad', width=1.5),
            name='Latent Mu_1',
            hovertemplate='<b>Mu_1:</b> %{y:.4f}<extra></extra>'
        ), secondary_y=True)
        
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['mu_7'],
            mode='lines', line=dict(color='#16a085', width=1.5),
            name='Latent Mu_7',
            hovertemplate='<b>Mu_7:</b> %{y:.4f}<extra></extra>'
        ), secondary_y=True)

    # 4. Thêm chú thích cho vùng Bull/Bear
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=10, color='rgba(0, 255, 0, 0.4)', symbol='square'),
        name='Bull (EMA5 > EMA12)'
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=10, color='rgba(255, 0, 0, 0.4)', symbol='square'),
        name='Bear (EMA5 < EMA12)'
    ))

    # 5. Tạo các vùng màu Bull/Bear
    shapes = []
    annotations = []
    
    phases = df['Phase'].values
    dates = df['Date'].dt.strftime('%Y-%m-%d').values
    
    start_idx = 0
    for i in range(1, len(phases)):
        if phases[i] != phases[i-1] or i == len(phases) - 1:
            prev_phase = phases[i-1]
            if prev_phase != 0:
                color = "rgba(46, 204, 113, 0.25)" if prev_phase == 1 else "rgba(231, 76, 60, 0.25)"
                label = "B" if prev_phase == 1 else "S"
                if i - start_idx > 10: label = "BULL" if prev_phase == 1 else "BEAR"
                
                shapes.append(dict(
                    type="rect", x0=dates[start_idx], x1=dates[i],
                    y0=0, y1=1, yref="paper",
                    fillcolor=color, line_width=0, layer="below"
                ))
                
                if i - start_idx > 5:
                    annotations.append(dict(
                        x=dates[start_idx + (i-start_idx)//2], y=1,
                        yref="paper", text=label, showarrow=False,
                        font=dict(size=9, color="black"), yanchor="bottom"
                    ))
            start_idx = i

    fig.update_layout(
        shapes=shapes,
        annotations=annotations,
        title={
            'text': "<b>Tương quan Giá FPT và Biến ẩn Latent Mu (EMA 5/12)</b>",
            'x': 0.5, 'xanchor': 'center'
        },
        xaxis_title="Thời gian",
        yaxis_title="Giá (VNĐ)",
        yaxis2_title="Giá trị Latent Mu",
        template="plotly_white",
        hovermode="x unified",
        height=850,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )

    fig.update_xaxes(rangeslider_visible=True)
    
    output_path = "results/market_phases_interactive.html"
    
    # Cấu hình tính năng zoom nâng cao và xuất ảnh chất lượng cao
    config = {
        'scrollZoom': True,        # Cho phép zoom bằng cuộn chuột
        'displaylogo': False,      # Ẩn logo plotly
        'responsive': True,        # Tự động thích ứng kích thước màn hình
        'toImageButtonOptions': {
            'format': 'png',       # Định dạng khi bấm vào icon máy ảnh
            'filename': 'FPT_Market_Analysis_HD',
            'height': 1080,
            'width': 1920,
            'scale': 2             # Nhân đôi mật độ điểm ảnh để cực nét
        }
    }
    
    fig.write_html(output_path, config=config)
    print(f"Thành công! Đã tích hợp Latent Mu và High-Res Zoom. Mở: {output_path}")

if __name__ == "__main__":
    price_csv = r"data/FPT Corp Stock Price History.csv"
    latent_csv = "results/latent_analysis_table.csv"
    
    if os.path.exists(price_csv):
        df_price = load_and_clean_data(price_csv)
        df = detect_phases(df_price)
        
        # Merge với latent data nếu có
        if os.path.exists(latent_csv):
            print(f"Đang tích hợp dữ liệu latent từ {latent_csv}...")
            df_latent = pd.read_csv(latent_csv)
            df_latent['Date'] = pd.to_datetime(df_latent['Date'])
            # Merge bằng Date
            df = pd.merge(df, df_latent[['Date', 'mu_1', 'mu_7']], on='Date', how='inner')
        
        plot_market_phases(df)
