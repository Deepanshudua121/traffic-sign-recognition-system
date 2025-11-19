from scripts.streamlit_app import run

if __name__ == "__main__":
    class Args:
        model_path = "models/best_model.h5"

    run(Args())
