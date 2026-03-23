from .build_features import FeatureBuilder
import logging

logging.basicConfig(level=logging.INFO) 


if __name__ == "__main__":
    logging.info("Building features...")
    builder = FeatureBuilder()
    df, feature_df, mean, std = builder.build("data/FPT Corp Stock Price History.csv")
    builder.save_features(feature_df, "data/FPT_features.csv")
    logging.info("Features built successfully!")
