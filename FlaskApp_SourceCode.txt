import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from flask import Flask, jsonify, request
import logging

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

# Load the models
dungeon_gan_model_path = 'Dungeon_GAN_Generator.h5'  # Update the path if needed
#dungeon_vae_encoder_model_path = 'Dungeon_VAE_Encoder.h5'  # Update the path if needed
#dungeon_vae_decoder_model_path = 'Dungeon_VAE_Decoder.h5'  # Update the path if needed
room_cgan_model_path = 'Room_CGAN_Generator.h5'  # Update the path if needed

dungeon_gan_model = load_model(dungeon_gan_model_path)
#dungeon_vae_encoder_model = load_model(dungeon_vae_encoder_model_path)
#dungeon_vae_decoder_model = load_model(dungeon_vae_decoder_model_path)
room_cgan_model = load_model(room_cgan_model_path)

# Define the dimensions and number of classes
noise_dim = 100
num_classes = 16

# Function to generate a single dungeon image using GAN
def generate_dungeon_gan(generator_model, noise_dim=100):
    noise = np.random.normal(0, 1, (1, noise_dim))
    generated_image = generator_model.predict(noise)
    return generated_image

# Function to generate a single dungeon image using VAE
def generate_dungeon_vae(encoder_model, decoder_model, latent_dim=100):
    noise = np.random.normal(0, 1, (1, latent_dim))
    latent_representation = encoder_model.predict(noise)
    generated_image = decoder_model.predict(latent_representation)
    return generated_image

# Function to generate a single room layout using Conditional GAN
def generate_room(generator_model, noise_dim=100, num_classes=16):
    noise = np.random.normal(0, 1, (1, noise_dim))
    gen_label = np.random.randint(0, num_classes, 1)
    gen_label_one_hot = to_categorical(gen_label, num_classes=num_classes)
    generated_image = generator_model.predict([noise, gen_label_one_hot])
    generated_image = 0.5 * generated_image + 0.5  # Rescale image to 0 - 1
    generated_image = generated_image.reshape((16, 16))  # Reshape to 16x16
    return generated_image

# Function to convert the generated image values to integers between 0 and 8
def convert_image_values(image):
    converted_image = np.clip(image * 8, 0, 8).astype(int)
    return converted_image

# Function to correct the room layout
def correct_room_layout(room):
    n = room.shape[0]
    corrected_room = np.zeros((n, n), dtype=int)
    corrected_room[0, :] = 1
    corrected_room[:, 0] = 1
    corrected_room[n-1, :] = 1
    corrected_room[:, n-1] = 1
    corrected_room[room == 7] = 7
    for i in range(1, n-1):
        for j in range(1, n-1):
            if room[i, j] == 7:
                corrected_room[i, j] = 7
            elif corrected_room[i, j] == 0:
                corrected_room[i, j] = room[i, j]
    return corrected_room

# Function to correct the dungeon image by thresholding
def correct_dungeon_image(image, threshold_value=0.5):
    corrected_image = np.where(image <= threshold_value, 0, 1)
    return corrected_image
    
# Convert image to list of integers
def image_to_int_list(image):
    int_list = image.flatten().astype(int).tolist()
    return int_list

# Create Flask app
app = Flask(__name__)

# Route to handle GET requests from Unity for Dungeon GAN
@app.route('/getDungeonGAN', methods=['GET'])
def generate_and_correct_dungeon_gan_image():
    try:
        generated_image = generate_dungeon_gan(dungeon_gan_model).reshape((8, 8))  # Assuming it's grayscale
        corrected_image = correct_dungeon_image(generated_image)
        int_list = image_to_int_list(corrected_image)
        response = {
            'status': 'Dungeon GAN image processed and corrected',
            'image_data': int_list
        }
        return jsonify(response)
    except Exception as e:
        logging.error(f"Error in generate_and_correct_dungeon_gan_image: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Route to handle GET requests from Unity for Dungeon VAE
@app.route('/getDungeonVAE', methods=['GET'])
def generate_and_correct_dungeon_vae_image():
    try:
        generated_image = generate_dungeon_vae(dungeon_vae_encoder_model, dungeon_vae_decoder_model).reshape((8, 8))  # Assuming it's grayscale
        corrected_image = correct_dungeon_image(generated_image)
        int_list = image_to_int_list(corrected_image)
        response = {
            'status': 'Dungeon VAE image processed and corrected',
            'image_data': int_list
        }
        return jsonify(response)
    except Exception as e:
        logging.error(f"Error in generate_and_correct_dungeon_vae_image: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/getRoomCGAN', methods=['GET'])
def generate_and_correct_room_layout():
    try:
        generated_image = generate_room(room_cgan_model).reshape((16, 16))  # Assuming it's grayscale
        converted_image = convert_image_values(generated_image)
        corrected_image = correct_room_layout(converted_image)
        int_list = image_to_int_list(corrected_image)
        response = {
            'status': 'Room layout processed and corrected',
            'room_data': int_list
        }
        return jsonify(response)
    except Exception as e:
        logging.error(f"Error in generate_and_correct_room_layout: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
