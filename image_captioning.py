import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Add, Dropout
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.translate.bleu_score import corpus_bleu
import pickle
from PIL import Image
import kagglehub

# Constants
IMAGE_SIZE = (299, 299)
MAX_CAPTION_LEN = 40
VOCAB_SIZE = 10000
EMBEDDING_DIM = 256
LSTM_UNITS = 512
DENSE_UNITS = 512
DROPOUT_RATE = 0.3
BATCH_SIZE = 64
EPOCHS = 20

class ImageCaptioningModel:
    def __init__(self):
        self.image_model = self.create_cnn()
        self.caption_model = self.create_lstm()
        self.tokenizer = None
        self.max_caption_length = None
        
    def create_cnn(self):
        """Create CNN feature extractor using InceptionV3"""
        inception = InceptionV3(weights='imagenet', include_top=False)
        inputs = inception.input
        outputs = inception.layers[-1].output
        return Model(inputs, outputs)
        
    def create_lstm(self):
        """Create LSTM decoder for caption generation"""
        # Image features input
        image_input = Input(shape=(2048,))
        image_dense = Dense(EMBEDDING_DIM, activation='relu')(image_input)
        image_dropout = Dropout(DROPOUT_RATE)(image_dense)
        
        # Caption input
        caption_input = Input(shape=(MAX_CAPTION_LEN,))
        caption_embedding = Embedding(VOCAB_SIZE, EMBEDDING_DIM)(caption_input)
        caption_lstm = LSTM(LSTM_UNITS, return_sequences=True)(caption_embedding)
        
        # Combine features
        combined = Add()([image_dropout, caption_lstm])
        outputs = Dense(VOCAB_SIZE, activation='softmax')(combined)
        
        return Model(inputs=[image_input, caption_input], outputs=outputs)
        
    def preprocess_data(self, image_dir, caption_file):
        """Preprocess images and captions"""
        # Validate inputs
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not os.path.exists(caption_file):
            raise FileNotFoundError(f"Caption file not found: {caption_file}")
            
        # Load and preprocess images
        image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]
        if not image_paths:
            raise ValueError(f"No images found in directory: {image_dir}")
            
        images = []
        for path in image_paths:
            try:
                img = load_img(path, target_size=IMAGE_SIZE)
                img = img_to_array(img)
                img = img / 255.0
                img = np.expand_dims(img, axis=0)
                img = self.image_model.predict(img, verbose=0)
                images.append(img)
            except Exception as e:
                raise ValueError(f"Error processing image {path}: {str(e)}")
        
        # Load and preprocess captions
        with open(caption_file, 'r') as f:
            captions = f.read().splitlines()
            
        # Initialize tokenizer
        self.tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(captions)
        
        # Convert captions to sequences
        sequences = self.tokenizer.texts_to_sequences(captions)
        self.max_caption_length = max(len(seq) for seq in sequences)
        padded_sequences = pad_sequences(sequences, maxlen=MAX_CAPTION_LEN, padding='post')
        
        # Save tokenizer for inference
        with open('tokenizer.pkl', 'wb') as f:
            pickle.dump(self.tokenizer, f)
            
        return np.array(images), padded_sequences
        
    def train(self, image_dir, caption_file):
        """Train the model"""
        # Validate inputs
        if not isinstance(BATCH_SIZE, int) or BATCH_SIZE <= 0:
            raise ValueError("BATCH_SIZE must be a positive integer")
        if not isinstance(EPOCHS, int) or EPOCHS <= 0:
            raise ValueError("EPOCHS must be a positive integer")
            
        # Preprocess data
        images, captions = self.preprocess_data(image_dir, caption_file)
        
        # Validate data shapes
        if len(images) == 0:
            raise ValueError("No valid images found for training")
        if len(captions) == 0:
            raise ValueError("No valid captions found for training")
            
        # Prepare training data
        X_images = images
        X_captions = captions[:, :-1]  # All words except last
        y_captions = captions[:, 1:]   # All words except first
        
        # Compile model
        try:
            self.caption_model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        except Exception as e:
            raise RuntimeError(f"Error compiling model: {str(e)}")
        
        # Create callbacks
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'image_captioning_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            patience=5,
            restore_best_weights=True
        )
        
        # Train model
        history = self.caption_model.fit(
            [X_images, X_captions],
            y_captions,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_split=0.2,
            callbacks=[checkpoint, early_stopping]
        )
        
        return history
        
    def generate_caption(self, image_path):
        """Generate caption for new image"""
        # Validate input path
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Validate tokenizer exists
        if not os.path.exists('tokenizer.pkl'):
            raise FileNotFoundError("Tokenizer file not found. Please train the model first.")
            
        try:
            # Load tokenizer
            with open('tokenizer.pkl', 'rb') as f:
                self.tokenizer = pickle.load(f)
                
            # Load and preprocess image
            try:
                img = load_img(image_path, target_size=IMAGE_SIZE)
                img = img_to_array(img)
                img = img / 255.0
                img = np.expand_dims(img, axis=0)
                img_features = self.image_model.predict(img, verbose=0)
            except Exception as e:
                raise ValueError(f"Error processing image: {str(e)}")
            
            # Initialize caption generation
            caption = '<start>'
            for _ in range(MAX_CAPTION_LEN):
                try:
                    # Convert current caption to sequence
                    sequence = self.tokenizer.texts_to_sequences([caption])[0]
                    sequence = pad_sequences([sequence], maxlen=MAX_CAPTION_LEN)
                    
                    # Predict next word
                    preds = self.caption_model.predict([img_features, sequence], verbose=0)
                    pred_word = np.argmax(preds[0, -1, :])
                    
                    # Convert word index to word
                    word = self.tokenizer.index_word.get(pred_word, '')
                    
                    # Stop if we predict the end token
                    if word == '<end>' or word == '':
                        break
                        
                    # Append word to caption
                    caption += ' ' + word
                except Exception as e:
                    raise RuntimeError(f"Error generating caption: {str(e)}")
            
            # Remove start token and return
            return caption.replace('<start>', '').strip()
            
        except Exception as e:
            raise RuntimeError(f"Error in caption generation process: {str(e)}")

if __name__ == "__main__":
    model = ImageCaptioningModel()
