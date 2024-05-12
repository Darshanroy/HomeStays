## Homestays: A Data-Driven Approach to House Price Prediction

**Project Description**

_Homestays is a data science project that empowers you to make informed decisions in the housing market. By leveraging the power of machine learning, 
Homestays analyzes a wide range of factors that influence house prices, aiming to predict their value with exceptional accuracy._


This project delves beyond simply providing a price tag. Homestays sheds light on the hidden patterns within the housing market,
giving you valuable insights into what truly drives value. With Homestays, you can:

- **Gain a competitive edge:** _Whether you're a buyer or seller, Homestays equips you with the knowledge to make strategic decisions._

- **Invest with confidence:** _Homestays empowers you to identify undervalued properties and make informed investment choices._   

- **Navigate the market with clarity:** _Homestays unravels the complexities of the housing market, allowing you to approach your next move with greater understanding._
                                                                                     

**Prerequisites:**

* Docker installed on your system. You can verify by running `docker -v` in your terminal. If not installed, refer to the official Docker documentation for installation instructions specific to your operating system [https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/).

**Steps:**

1. **Clone the project repository:**

   Open your terminal or command prompt and navigate to the directory where you want to clone the project. Then, run the following command to clone the repository from GitHub:

   ```bash
   git clone https://github.com/Darshanroy/HomeStays.git
   ```

   This will download the project files to a local directory named "HomeStays".

2. **Install dependencies (requirements.txt):**

   Navigate into the downloaded project directory:

   ```bash
   cd HomeStays
   ```

   Then, install the required Python libraries listed in the `requirements.txt` file using pip:

   ```bash
   pip install -r requirements.txt
   ```

   This will download and install the necessary libraries for your project to run.

3. **Train the model:**

   Run the following command to train your machine learning model using the `training.py` script:

   ```bash
   python training.py
   ```

   This script will likely take some time to complete depending on the size and complexity of your dataset. The training process might involve data preparation, model training, and evaluation.

4. **Run the application:**

   Once the model is trained, start the web application using the following command:

   ```bash
   docker run -p 5000:5000 flaskapp:v1
   ```

   This command will:

   * Run a Docker container based on the image named `flaskapp:v1`. This image likely contains your application code and the trained model.
   * Publish the container's port 5000 to your host machine's port 5000. This allows you to access the application running inside the container through your web browser.

5. **Access the application:**

   Open your web browser and navigate to http://localhost:5000 (or http://127.0.0.1:5000 if needed). This should launch your Homesatys house price prediction application.

