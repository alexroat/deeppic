import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2


print("MPS available:",torch.backends.mps.is_available())
print("MPS built:",torch.backends.mps.is_built())
device = torch.device("mps")
print(device)
dtype = torch.float


def create_model():
    model = nn.Sequential(
        nn.Linear(2, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 3),
        nn.Sigmoid()
    )
    return model

# Titolo fisso
st.title("Neural Network Image Prediction")
# Due placeholder fissi per l'immagine
# Crea due colonne per le immagini affiancate
col1, col2 = st.columns(2)
image_placeholder1 = col1.empty()
image_placeholder2 = col2.empty()
report_placeholder=st.empty()

# Funzione per addestrare il modello
def train_model(model, X, Y, num_epochs=1000):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        report=f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}'
        st.write(report)
        print(report)

# Funzione per visualizzare l'immagine originale e quella stimata
def show_images(original, predicted):
    st.image(original, caption='Original Image', use_column_width=True)
    st.image(predicted, caption='Predicted Image', use_column_width=True)


def load_image_as_tensor(file_name,scale=0.3):
    # Carica l'immagine utilizzando OpenCV
    image = cv2.imread(file_name)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Converte il formato dei colori da BGR a RGB
    
    # Ottieni le dimensioni dell'immagine
    h, w, _ = image.shape
    
    image=cv2.resize(image,(int(w*scale),int(h*scale)))
    return torch.tensor(image, dtype=torch.float32)  # (3, h, w)

def tensor_to_image(tensor):
    # Assicurati che i valori siano nel range 0-255
    img_array = np.clip(tensor.cpu().detach().numpy(), 0, 255).astype(np.uint8)

    # Converte l'array NumPy in un'immagine nel formato corretto (BGR) utilizzando OpenCV
    return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)


def generate_coords_tensor(w,h):
     # Genera le coordinate dei pixel utilizzando meshgrid e linspace
    x_coords = np.linspace(0, w - 1, w)
    y_coords = np.linspace(0, h - 1, h)
    xv, yv = np.meshgrid(x_coords, y_coords)
    
    # Concatena le coordinate dei pixel in un tensore (h, w, 2)
    pixel_coords = np.stack([xv, yv], axis=-1)
    return torch.tensor(pixel_coords, dtype=torch.float32)  # (h, w, 2)
    

def load_image_and_coordinates(file_name,scale=0.3):
    image=load_image_as_tensor(file_name=file_name,scale=scale)
    # Ottieni le dimensioni dell'immagine
    h, w, _ = image.shape
    coords=generate_coords_tensor(w,h)
    return image,coords
    


def img2batch(img):
    return img.view(-1, img.shape[-1]).unsqueeze(0)

def batch2img(batch,w,h):
    return torch.reshape(batch, (h,w, 3))





img,coords=load_image_and_coordinates("image.png",scale=0.05)

h,w,_=img.shape

coords[:,:,0]/=w
coords[:,:,1]/=h
img[:]/=255.


print(img.shape,coords.shape)
print(img2batch(img).shape,img2batch(coords).shape)



X=img2batch(coords).to(device)
Y=img2batch(img).to(device)


num_epochs=10000
model=create_model().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, Y)
    loss.backward()
    optimizer.step()
    
    decoded=batch2img(outputs,w,h)
    imout=tensor_to_image(decoded*255)
    imin=tensor_to_image(img*255)
    image_placeholder1.image(imin, caption='input', use_column_width=True)
    image_placeholder2.image(imout, caption='output', use_column_width=True)

    
    report=f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}'
    report_placeholder.write(report)


