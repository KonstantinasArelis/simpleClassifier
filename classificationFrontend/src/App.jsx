import { useState, useEffect, useRef } from 'react'
import './App.css'
import FileInput from './FileInput.jsx'

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imageLabel, setImageLabel] = useState(null);
  const [labelProbability, setLabelProbability] = useState(null);
  const imageLabelref = useRef(null);
  const labelProbabilityref = useRef(null);


  const handleFileSelected = (file) => {
    console.log('Selected file:', file);
    setSelectedImage(file);
    // You can now work with the selected file (e.g., upload it)
  };

  useEffect(() => {
    if(selectedImage !== null){
      handleUpload();
    }
  }, [selectedImage])

  useEffect(() => {
    if(imageLabelref.current){
      imageLabelref.current.classList.add("imageLabel");
      
      setTimeout(() => {
        imageLabelref.current.classList.remove("imageLabel");
      }, 2000)
    }
  }, [imageLabel])

  useEffect(() => {
    setTimeout(() => {
      if(labelProbabilityref.current){
              labelProbabilityref.current.classList.add("labelProbability");
              setTimeout(() => {
                labelProbabilityref.current.classList.remove("labelProbability");
              }, 2000)
            }
    },500)
  }, [labelProbability])
  
  const handleUpload = async () => {
    if (!selectedImage) {
      alert('Please select an image first.');
      return;
    }

    const formData = new FormData();
    formData.append('image', selectedImage); // 'image' is the field name the API expects
    try {
      // Make the API call using fetch or another HTTP library (like axios)
      const response = await fetch('http://localhost:5000/classify', { // Replace '/api/upload' with your actual API endpoint
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        console.log('Image uploaded successfully:', data.predicted_class);
        console.log('Image uploaded successfully:', data.confidence);
        // Optionally reset the selected image state
        setImageLabel(data.predicted_class);
        setLabelProbability((data.confidence * 100).toFixed(2));
      } else {
        console.error('Error uploading image:', data);
        alert('Error uploading image.');
      }
    } catch (error) {
      console.error('Network error:', error);
      alert('Network error occurred.');
    }
  };

  return (
    <>
      {imageLabel && (
        <div>
          <h1 ref={imageLabelref}>Its a {imageLabel}!</h1>
          <h3 ref={labelProbabilityref}>{labelProbability} %</h3>
        </div>
      )}
      {!imageLabel && (
        <h1>Image Classifier</h1>
      )}
      <div className="card">
        
        <div>
         <FileInput onFileSelect={handleFileSelected}></FileInput>
        </div>
        {selectedImage && (
          <div>
          <img className='selectedImage' src={URL.createObjectURL(selectedImage)} alt="Selected"></img>
        </div>
        )}
        {!selectedImage && (<div className='imagePlaceholder'/>)}
      </div>

      <footer>Konstantinas Arelis</footer>
    </>
  )
}

export default App
