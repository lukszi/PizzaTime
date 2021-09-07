# PIZZA TIME

This is a Telegram bot, that tells you when the pizza man is at your Floor

To achieve this, we take photos of the elevator labels at the OPH-Student dorm, extract the number images and attempt to extract the floor the elevator is currently at.
This is done using openCV for contouring, and using a Keras CNN to classify the contoured numbers.