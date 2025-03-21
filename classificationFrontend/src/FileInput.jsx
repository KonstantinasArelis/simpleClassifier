import React, { useState, useRef } from 'react';
import './FileInput.css'

const FileInput = ({ onFileSelect }) => {
    const [fileName, setFileName] = useState('No file chosen');
    const [isDragging, setIsDragging] = useState(false);
    const fileInputRef = useRef(null);
    const dropZoneRef = useRef(null); // Ref for the drop zone element

    const handleButtonClick = () => {
        fileInputRef.current.click(); // Trigger the hidden file input
    };

    const handleDragOver = (event) => {
        event.preventDefault(); // Prevent default browser behavior to allow drop
        setIsDragging(true);
      };
    
      const handleDragEnter = (event) => {
        setIsDragging(true);
      };
    
      const handleDragLeave = (event) => {
        setIsDragging(false);
      };
    
      const handleDrop = (event) => {
        event.preventDefault(); // Prevent default browser behavior
        setIsDragging(false);
        const droppedFiles = event.dataTransfer.files;
        if (droppedFiles && droppedFiles.length > 0) {
          setFileName(Array.from(droppedFiles).map(file => file.name).join(', '));
          if (onFileSelect) {
           onFileSelect(droppedFiles[0]); // Pass the dropped files
          }
        }
      };

      return (
        <div className={"fileUpload"}>
          <div
            ref={dropZoneRef}
            className={`${"dropZone"} ${isDragging ? "isDragging" : ''}`}
            onDragOver={handleDragOver}
            onDragEnter={handleDragEnter}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <button className={"customButton"} onClick={handleButtonClick}>
              Choose File
            </button>
            <span className={"fileName"}>{fileName}</span>
          </div>
          <input
            type="file"
            ref={fileInputRef}
            className={"hiddenFileInput"}
            onChange={"handleFileChange"}
            multiple // Allow multiple file selection for drag and drop
          />
        </div>
      );
}

export default FileInput;