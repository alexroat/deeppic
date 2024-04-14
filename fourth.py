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




class FourierLayer(nn.Module):
    def __init__(self, input_dim,output_dim):
        super(FourierLayer, self).__init__()
        self.w=32*torch.pi*torch.randn((input_dim,output_dim)).to(device)
        self.p=32*torch.pi*torch.randn(output_dim).to(device)
    def forward(self, x):
        y=torch.cos(torch.matmul(x,self.w)+self.p)
        return y
               

def create_model():
    model = nn.Sequential(
        FourierLayer(2,1000),
        nn.Linear(1000, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.Linear(100, 3),
        nn.Sigmoid()
    )
    return model

def subsample(tensors,dim,n):
    idxs = torch.randperm(tensors[0].size(1))[:n]
    return [t[:,idxs,:] for t in tensors]


# Titolo fisso
st.title("Neural Network Image Prediction")
# Due placeholder fissi per l'immagine
# Crea due colonne per le immagini affiancate
col1, col2 = st.columns(2)
image_placeholder1 = col1.empty()
image_placeholder2 = col2.empty()
report_placeholder=st.empty()



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



def chunked_inference(model,X,dim=0,size=1000):
    print(X.shape)
    sub_tensors = torch.split(X, size, dim=dim)
    
    print(sub_tensors[0].shape)
    with torch.no_grad():
        outputs=[]
        for i,st in enumerate(sub_tensors):
            o=model(st)
            outputs.append(o)
            output=torch.cat(outputs, dim=dim)
        return output


img,coords=load_image_and_coordinates("image.png",scale=0.5)

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
    
    Xs,Ys=subsample((X,Y),1,9000)
    
    outputs = model(Xs)
    loss = criterion(outputs, Ys)
    loss.backward()
    optimizer.step()
    
    report=f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}'
    report_placeholder.write(report)
    print(report)
    
    
    
    #outputs = model(X)
    
    if epoch%100:
        continue
    
    outputs=chunked_inference(model,X,dim=1,size=1000)
    decoded=batch2img(outputs,w,h)
    imout=tensor_to_image(decoded*255)
    imin=tensor_to_image(img*255)
    image_placeholder1.image(imin, caption='input', use_column_width=True)
    image_placeholder2.image(imout, caption='output', use_column_width=True)

    
    report=f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}'
    report_placeholder.write(report)


