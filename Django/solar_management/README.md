# solarVert

**solarVert** is a solar panel management system that provides various functionalities such as anomaly detection, power generation predictions, power usage predictions, and maximum power point tracking (MPPT). It uses a [Flutter](https://github.com/shubhikaj/solarVert) frontend and a Django backend integrated with machine learning and deep learning models to make predictions and analyze solar panel performance.

## Features

- Weather forecast advice for power saving
- Check if a solar panel is defective
- View power input and usage data
- Anomaly detection for panel performance
- Power generation prediction
- Power usage prediction

## Tech Stack

- **Frontend**: Flutter
- **Backend**: Django, Django REST Framework
- **Machine Learning Models**: TensorFlow, scikit-learn
- **Database**: SQLite3
- **Deployment**: Linux (Fedora 40)

## Installation

### Prerequisites

- Python 3.x
- pip (Python package installer)
- Flutter SDK
- Android Studio or any other compatible IDE for Flutter development

### Backend Setup

1. Clone the repository:

```bash
git clone https://github.com/arion52/solarVert.git
cd solarVert
```

2. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required Python packages:

```bash
pip install -r requirements.txt
```

4. Set up the database:

```bash
python manage.py migrate
```

5. Run the development server:

```bash
python manage.py runserver
```

### Frontend Setup

1. Navigate to the `flutter` directory:

```bash
cd flutter
```

2. Run the Flutter app:

```bash
flutter run
```

Ensure you have the necessary Android/iOS device/emulator set up.

## API Endpoints

- **Weather Advice**: `/weather_advice/`
- **Check Panel Status**: `/check_panel/<panel_id>/`
- **Power Data**: `/power_data/<sensor_id>/`
- **Predict Anomaly**: `/predict_anomaly/`
- **Predict Generation**: `/predict_generation/`
- **Predict Usage**: `/predict_usage/`
- **Post Sensor Data**: `/post_sensor_data/`

## License

This project is licensed under the MPL 2.0 License.
