import { useState, useEffect, useRef } from 'react'
import './App.css'
import FileInput from './FileInput.jsx'
import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Form from 'react-bootstrap/Form';
import { BsArrowRight } from "react-icons/bs";
import { FaSpinner } from 'react-icons/fa';


function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imageLabel, setImageLabel] = useState(null);
  const [labelProbability, setLabelProbability] = useState(null);
  const [imageSegmentationMask, setImageSegmentationMask] = useState(null);
  const [imageSegmentationLabelData, setImageSegmentationLabelData] = useState(null);
  const [mode, setMode] = useState("segmentation");
  const [isLoading, setIsLoading] = useState(false);
  const imageLabelref = useRef(null);
  const labelProbabilityref = useRef(null);

  const handleModeSwitch = () => {
    setImageLabel(null);
    setSelectedImage(null);
    setImageSegmentationLabelData(null);
    setImageSegmentationMask(null);
    if (mode === "clasification") {
      setMode("segmentation");
    } else {
      setMode("clasification");
    }
  }

  const handleFileSelected = (file) => {
    console.log('Selected file:', file);
    setSelectedImage(file);
    // You can now work with the selected file (e.g., upload it)
  };

  useEffect(() => {
    if (selectedImage !== null) {
      if (mode === "clasification") {
        handleUpload();
      } else if (mode === "segmentation") {
        handleUploadSegmentation();
      }
    }
  }, [selectedImage])

  useEffect(() => {
    if (imageLabelref.current) {
      imageLabelref.current.classList.add("imageLabel");

      setTimeout(() => {
        imageLabelref.current.classList.remove("imageLabel");
      }, 2000)
    }
  }, [imageLabel])

  useEffect(() => {
    setTimeout(() => {
      if (labelProbabilityref.current) {
        labelProbabilityref.current.classList.add("labelProbability");
        setTimeout(() => {
          labelProbabilityref.current.classList.remove("labelProbability");
        }, 2000)
      }
    }, 500)
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

  const handleUploadSegmentation = async () => {
    if (!selectedImage) {
      alert('Please select an image first.');
      return;
    }

    const formData = new FormData();
    formData.append('image', selectedImage); // 'image' is the field name the API expects
    try {
      setImageSegmentationLabelData(null);
      setImageSegmentationMask(null);
      setIsLoading(true);
      // Make the API call using fetch or another HTTP library (like axios)
      const response = await fetch('http://localhost:5000/segment', { // Replace '/api/upload' with your actual API endpoint
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        console.log('Image uploaded successfully:', data.segmentation_mask);

        setImageSegmentationMask(data.segmentation_mask);
        setImageSegmentationLabelData(data.detected_classes);
      } else {
        console.error('Error uploading image:', data);
        alert('Error uploading image.');
      }
    } catch (error) {
      console.error('Network error:', error);
      alert('Network error occurred.');
    } finally{
      setIsLoading(false);
    }
  };

  if (mode === 'clasification') {
    return (
      <>
        <header style={{ display: 'flex', position: 'fixed', left: '80%', top: '5%' }}>


          <p style={{ marginTop: '8px' }}>Segmentation</p>
          <Form.Check
            className='fs-3'
            type="switch"
            id="custom-switch"
            onClick={handleModeSwitch}
          />

          <p style={{ marginTop: '8px' }}>Clasification</p>

        </header>


        {imageLabel && (
          <div>
            <h1 ref={imageLabelref}>Its a {imageLabel}!</h1>
            <h3 ref={labelProbabilityref}>{labelProbability} %</h3>
          </div>
        )}
        {!imageLabel && (
          <h1>Image {mode}</h1>
        )}
        <div>

          <div>
            <FileInput onFileSelect={handleFileSelected}></FileInput>
          </div>
          {selectedImage && (
            <div>
              <img className='selectedImage' src={URL.createObjectURL(selectedImage)} alt="Selected"></img>
            </div>
          )}
          {!selectedImage && (<div className='imagePlaceholder' />)}
        </div>

          



        <footer>Konstantinas Arelis</footer>
      </>
    )
  } else if (mode === 'segmentation') {
    return (
      <>
        <header style={{ display: 'flex', position: 'fixed', left: '80%', top: '5%' }}>


          <p style={{ marginTop: '8px' }}>Segmentation</p>
          <Form.Check
            className='fs-3'
            type="switch"
            id="custom-switch"
            onClick={handleModeSwitch}
          />

          <p style={{ marginTop: '8px' }}>Clasification</p>

        </header>

        <h1>Image {mode}</h1>

        <div>

          <div>
            <FileInput onFileSelect={handleFileSelected}></FileInput>
          </div>

          <Container>
            <Row style={{display: 'flex', justifyContent: 'center', alignItems: 'center'}}>
              <Col>
                {selectedImage && (
                  <div>
                    <img className='selectedImage' src={URL.createObjectURL(selectedImage)} alt="Selected"></img>
                  </div>
                )}
                {!selectedImage && (<div className='imagePlaceholder' />)}
              </Col>
              <Col>
                <BsArrowRight size={"200px"}/>

              </Col>
              <Col>
                {imageSegmentationMask && imageSegmentationLabelData && (
                  <div>
                    <img src={imageSegmentationMask}></img>

                    {imageSegmentationLabelData.map((d) => {
                      return (
                        <div key={d.index}>
                          <Container style={{margin: '10px'}}>
                            <Row>
                              <Col>
                                <div style={{aspectRatio: '1', width: '30px', backgroundColor: `rgb(${d.color[0]}, ${d.color[1]}, ${d.color[2]})` }}>
                                  
                                </div>
                              </Col>
                              <Col>
                                {d.name}
                              </Col>
                            </Row>
                          </Container>
                        </div>
                      )
                    })}
                  </div>
                )}
                {! isLoading && !imageSegmentationMask && !imageSegmentationLabelData && (
                  <div className='imagePlaceholder' />
                )}
                {isLoading && (
                  <FaSpinner size={"200px"} className="icon-spin" />
                )}
              </Col>
            </Row>
          </Container>
        </div>
        <footer>Konstantinas Arelis</footer>
      </>
    )
  }
}

export default App
