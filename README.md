# 🧠 AI-Powered Brain Tumor Detection System

## 🚨 Problem
• Manual MRI analysis is slow and depends on expert radiologists  
• Early-stage tumors are difficult to detect accurately  
• No fast automated tool for instant classification of MRI scans  
• High workload in medical imaging departments  

## ✅ Solution
• Built an AI-based Brain Tumor Classification system using PyTorch  
• Used ResNet18 deep learning model trained on MRI images  
• Developed a Streamlit web app for real-time image upload and prediction  
• Automated preprocessing using torchvision transforms  
• Displayed prediction results with confidence scores and charts  

## 📈 Output
• Upload MRI image and get instant prediction  
• Detects tumor type: glioma, meningioma, pituitary, or no tumor  
• Shows confidence percentage for each class  
• Interactive bar chart for better understanding  
• Fast and simple medical AI interface  

## 🛠 Tools Used
• Python  
• PyTorch (Deep Learning)  
• ResNet18 (CNN Model)  
• Streamlit (Web App UI)  
• Torchvision (Image Processing)  
• Pandas (Data Handling)  
• PIL (Image Loading)  
• OS Module (File Path Handling)  

## 🧠 Model Working
• MRI image uploaded by user  
• Image resized to 224x224  
• Converted into tensor format  
• Passed through trained ResNet18 model  
• Softmax applied to get probabilities  
• Final class selected with highest confidence  

## 🌍 Real-Life Applications
• Medical MRI scan assistance  
• Radiology support systems  
• Early tumor screening tools  
• Healthcare AI research projects  
• Educational deep learning projects  

## 🏁 Conclusion
This system helps in fast and automated brain tumor classification from MRI scans using deep learning, reducing manual effort and improving early diagnosis support.
