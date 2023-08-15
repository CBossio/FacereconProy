import cv2
import matplotlib.pyplot as plt

try:
    imagePath = '20221018_152041.jpeg'
    
    img = cv2.imread(imagePath)
    
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    
    faces = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
    
    # Guardar la imagen con las detecciones
    output_path = 'output_image.jpg'
    cv2.imwrite(output_path, img)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(20, 10))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

except Exception as e:
    print("Error:", e)
