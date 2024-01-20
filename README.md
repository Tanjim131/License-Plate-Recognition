# License Plate Recognition App

This program recognizes the license plate number from a user provided image (if one exists).

## Setup

To set up this project, first clone the repository. Then create a virtual environment and activate it:

```bash
python -m venv lprenv
source lprenv/bin/activate  # On Windows use `lprenv\Scripts\activate`
```

It will install the required packages inside the virtual environment.

## Running the program

1. First run the following command to start a production server at port `8080` (default configuration). The server will also listen to all IP addresses on the host by default.

```bash
waitress-serve app.main:app
```

2. Once the server is up and running, you can send a request to the server with an image file:

```bash
curl -X POST -F "image=sample-test-car.png" http://127.0.0.1:8080/upload
```

The above sample request sends the image `sample-test-car.png` as the payload to the `/upload` POST API.

The API will return the following response:

```bash
whole_image_plate_number: "VIPER"
```

The Jupyter Notebook file `/app/license-plate-recognition.ipynb` delineates the dataset, methodology, configuration parameters and experimental results.
