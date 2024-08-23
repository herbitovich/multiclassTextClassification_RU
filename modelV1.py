import pandas as pd
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
# Load the dataset
data = pd.read_csv('data.csv')
# Split the data into training and validation sets
from sklearn.model_selection import train_test_split
train_texts, val_texts, train_labels, val_labels = train_test_split(data['text'], data['category'], random_state=42, test_size=0.2)
# Load the pre-trained tokenizer
tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased', from_pt=True, model_max_length=256)
# Tokenize the texts
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True)
# Convert the labels to numpy arrays
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=6)
val_labels = tf.keras.utils.to_categorical(val_labels, num_classes=6)
# Create the model
model = TFBertForSequenceClassification.from_pretrained('DeepPavlov/rubert-base-cased', num_labels=6, from_pt=True)
# Compile the model
print("prepared the data, initialized the models")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
metrics=['accuracy'])
print("compiled the model")
# Create datasets
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels)).batch(16)
val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), val_labels)).batch(16)
# Train the model
print("training the model")
history = model.fit(train_dataset, validation_data=val_dataset, epochs=5, batch_size=16)
# Evaluate the model
print("evaluating the model")
test_loss, test_acc = model.evaluate(val_dataset)
print(f'Test accuracy: {test_acc:.2f}')
# Use the model for predictions
predictions = model.predict(val_dataset)
predicted_classes = tf.argmax(predictions.logits, axis=1)
print(predicted_classes)
# Save the model
model.save_pretrained('rubert-base-cased-multiclass')