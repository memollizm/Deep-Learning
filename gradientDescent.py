# Numpy kütüphanesinin içeri aktarılması
import numpy as np

"""
- X ve y isimli iki numpy dizisi
- X, 4 örnekten oluşan bir giriş dizisi
- y, XOR işleminin beklenen çıktısı 
"""
X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
y = np.array([[0,1,1,0]]).T

"""
- alpha = Öğrenme oranı
- hidden_dim = Gizli katmanın boyutu
"""
alpha,hidden_dim = (0.5,4)

synapse_0 = 2*np.random.random((3,hidden_dim)) - 1 #Sinir ağının katmanları
synapse_1 = 2*np.random.random((hidden_dim,1)) - 1 #Sinir ağının katmanları


#Sinir ağını eğitmek için oluşturulan döngü [60000 iterasyon]
for j in range(60000):
    #Aktivasyon fonksiyonu olarak Sigmoid fonksiyonu kullanılır (1/(1+np.exp(-x)))
    layer_1 = 1/(1+np.exp(-(np.dot(X,synapse_0)))) 
    layer_2 = 1/(1+np.exp(-(np.dot(layer_1,synapse_1))))

    #Geri yayılım (backpropagation) işlemi
    layer_2_delta = (layer_2 - y)*(layer_2*(1-layer_2))
    layer_1_delta = layer_2_delta.dot(synapse_1.T) * (layer_1 * (1-layer_1))

    #Ağırlıkların güncellenmesi
    #Bu işlem sinir ağının öğrenmesini sağlar
    synapse_1 -= (alpha * layer_1.T.dot(layer_2_delta))
    synapse_0 -= (alpha * X.T.dot(layer_1_delta))

print("synapse_0: ",synapse_0)
print("synapse_1: ",synapse_1)