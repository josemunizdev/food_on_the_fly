"""Gradio app for Food on the Fly delivery time prediction."""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import joblib
import pandas as pd

MODEL_PATH = Path(__file__).parent / "pipeline.joblib"
_pipeline = joblib.load(MODEL_PATH)


def predict(
    age: float,
    ratings: float,
    restaurant_lat: float,
    restaurant_lon: float,
    delivery_lat: float,
    delivery_lon: float,
    order_date: str,
    time_ordered: str,
    weather: str,
    traffic: str,
    vehicle_condition: int,
    order_type: str,
    vehicle_type: str,
    multiple_deliveries: float,
    festival: str,
    city: str,
) -> str:
    row = pd.DataFrame(
        [
            {
                "Delivery_person_Age": age,
                "Delivery_person_Ratings": ratings,
                "Restaurant_latitude": restaurant_lat,
                "Restaurant_longitude": restaurant_lon,
                "Delivery_location_latitude": delivery_lat,
                "Delivery_location_longitude": delivery_lon,
                "Order_Date": order_date,
                "Time_Orderd": time_ordered,
                "Weather_conditions": weather,
                "Road_traffic_density": traffic,
                "Vehicle_condition": vehicle_condition,
                "Type_of_order": order_type,
                "Type_of_vehicle": vehicle_type,
                "multiple_deliveries": multiple_deliveries,
                "Festival": festival,
                "City": city,
            }
        ]
    )
    mins = max(0.0, round(float(_pipeline.predict(row)[0]), 1))

    if mins < 20:
        label, color = "Fast", "#22c55e"
    elif mins < 35:
        label, color = "Normal", "#f59e0b"
    else:
        label, color = "Slow", "#ef4444"

    badge = f'<b style="color:{color}">{label}</b>'
    return (
        f'<div style="text-align:center;padding:24px 0">'
        f'<div style="font-size:3rem;font-weight:700;color:{color}">{mins} min</div>'
        f'<div style="font-size:1rem;margin-top:4px">{badge}</div>'
        f'<div style="font-size:0.85rem;color:#6b7280;margin-top:12px">'
        f"Estimated delivery time</div></div>"
    )


EXAMPLES = [
    [
        29,
        4.5,
        19.076,
        72.878,
        19.090,
        72.860,
        "15-03-2022",
        "12:30",
        "Sunny",
        "Medium",
        2,
        "Meal",
        "motorcycle",
        0,
        "No",
        "Urban",
    ],
    [
        35,
        3.8,
        12.972,
        77.595,
        12.985,
        77.610,
        "20-02-2022",
        "19:00",
        "Cloudy",
        "High",
        1,
        "Snack",
        "scooter",
        1,
        "No",
        "Metropolitian",
    ],
    [
        42,
        5.0,
        28.614,
        77.209,
        28.630,
        77.220,
        "05-03-2022",
        "13:00",
        "Windy",
        "Jam",
        0,
        "Buffet",
        "motorcycle",
        2,
        "Yes",
        "Urban",
    ],
]

with gr.Blocks(title="Food on the Fly") as demo:
    gr.Markdown(
        """
        # 🛵 Food on the Fly
        ### Delivery Time Predictor
        Fill in the order details below to get an estimated delivery time.
        """
    )

    with gr.Row():
        # ── Left column ────────────────────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("#### 👤 Delivery Person")
            with gr.Row():
                age = gr.Number(label="Age", value=29, minimum=18, maximum=70)
                ratings = gr.Number(
                    label="Rating (1–6)", value=4.5, minimum=1, maximum=6
                )

            gr.Markdown("#### 📍 Location")
            with gr.Row():
                restaurant_lat = gr.Number(label="Restaurant lat", value=19.076)
                restaurant_lon = gr.Number(label="Restaurant lon", value=72.878)
            with gr.Row():
                delivery_lat = gr.Number(label="Delivery lat", value=19.090)
                delivery_lon = gr.Number(label="Delivery lon", value=72.860)

            gr.Markdown("#### 🛍️ Order")
            with gr.Row():
                order_date = gr.Textbox(label="Date (DD-MM-YYYY)", value="15-03-2022")
                time_ordered = gr.Textbox(label="Time (HH:MM)", value="12:30")
            with gr.Row():
                order_type = gr.Dropdown(
                    label="Order type",
                    choices=["Meal", "Snack", "Drinks", "Buffet"],
                    value="Meal",
                )
                vehicle_type = gr.Dropdown(
                    label="Vehicle",
                    choices=["motorcycle", "scooter", "electric_scooter", "bicycle"],
                    value="motorcycle",
                )
            with gr.Row():
                multiple_deliveries = gr.Slider(
                    label="Simultaneous deliveries",
                    minimum=0,
                    maximum=3,
                    step=1,
                    value=0,
                )
                festival = gr.Radio(
                    label="Festival day?", choices=["Yes", "No"], value="No"
                )

        # ── Right column ───────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("#### 🌤️ Conditions")
            weather = gr.Dropdown(
                label="Weather",
                choices=["Sunny", "Cloudy", "Windy", "Fog", "Sandstorms", "Stormy"],
                value="Sunny",
            )
            traffic = gr.Dropdown(
                label="Traffic density",
                choices=["Low", "Medium", "High", "Jam"],
                value="Medium",
            )
            city = gr.Dropdown(
                label="City type",
                choices=["Urban", "Metropolitian", "Semi-Urban"],
                value="Urban",
            )
            vehicle_condition = gr.Slider(
                label="Vehicle condition (0–3)", minimum=0, maximum=3, step=1, value=2
            )

            gr.Markdown("#### ⏱️ Prediction")
            output = gr.HTML()
            predict_btn = gr.Button(
                "Predict delivery time", variant="primary", size="lg"
            )

    gr.Markdown("#### 📋 Examples")
    gr.Examples(
        examples=EXAMPLES,
        inputs=[
            age,
            ratings,
            restaurant_lat,
            restaurant_lon,
            delivery_lat,
            delivery_lon,
            order_date,
            time_ordered,
            weather,
            traffic,
            vehicle_condition,
            order_type,
            vehicle_type,
            multiple_deliveries,
            festival,
            city,
        ],
        outputs=output,
        fn=predict,
        cache_examples=False,
    )

    predict_btn.click(
        fn=predict,
        inputs=[
            age,
            ratings,
            restaurant_lat,
            restaurant_lon,
            delivery_lat,
            delivery_lon,
            order_date,
            time_ordered,
            weather,
            traffic,
            vehicle_condition,
            order_type,
            vehicle_type,
            multiple_deliveries,
            festival,
            city,
        ],
        outputs=output,
    )

if __name__ == "__main__":
    demo.launch()
