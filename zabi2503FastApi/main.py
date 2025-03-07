from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn

from src.ML.ML_inference import load_model, load_scaler, get_data, predict_close_price


app = FastAPI(title="Crypto Forecast")

templates = Jinja2Templates(directory="templates")

btc_linear_model = load_model("src/ML/artifacts/model/btcusdt_1d_linear_model.pkl")
btc_xgb_model = load_model("src/ML/artifacts/model/btcusdt_1d_xgboost_model.pkl")
btc_lgbm_model = load_model("src/ML/artifacts/model/btcusdt_1d_lgbm_model.pkl")
btc_scaler = load_scaler("src/ML/artifacts/scaler/btcusdt_1d_scaler.pkl")

eth_linear_model = load_model("src/ML/artifacts/model/ethusdt_1d_linear_model.pkl")
eth_xgb_model = load_model("src/ML/artifacts/model/ethusdt_1d_xgboost_model.pkl")
eth_lgbm_model = load_model("src/ML/artifacts/model/ethusdt_1d_lgbm_model.pkl")
eth_scaler = load_scaler("src/ML/artifacts/scaler/ethusdt_1d_scaler.pkl")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict_price(
    request: Request,
    model: str = Form(...),
    crypto: str = Form(...),
    date: str = Form(...)
):

    prediction = None

    if crypto.lower() == "bitcoin":
        input_data = get_data(symbol="BTCUSDT", end_date=date)
        scaler = btc_scaler
        if model == "linear_regression":
            prediction = predict_close_price(btc_linear_model, scaler, input_data)
        elif model == "xgboost":
            prediction = predict_close_price(btc_xgb_model, scaler, input_data)
        elif model == "lightgbm":
            prediction = predict_close_price(btc_lgbm_model, scaler, input_data)

    elif crypto.lower() == "ethereum":
        input_data = get_data(symbol="ETHUSDT", end_date=date)
        scaler = eth_scaler
        if model == "linear_regression":
            prediction = predict_close_price(eth_linear_model, scaler, input_data)
        elif model == "xgboost":
            prediction = predict_close_price(eth_xgb_model, scaler, input_data)
        elif model == "lightgbm":
            prediction = predict_close_price(eth_lgbm_model, scaler, input_data)

    context = {
        "request": request,
        "prediction": round(prediction, 2) if prediction is not None else None,
        "crypto": crypto,
        "model": model,
        "date": date,
        "error": None if prediction is not None else "Invalid selection or missing data.",
    }
    return templates.TemplateResponse("index.html", context)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
