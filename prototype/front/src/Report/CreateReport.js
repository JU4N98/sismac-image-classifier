import React, { useState, resolve} from "react";
import {postReport} from "../ApiManager/ApiManager"

const ReportForm = () => {
    const [name, setName] = useState('');
    const [description, setDescription] = useState('');
    const [images, setImages] = useState([]);
  
    const handleImageUpload = (event) => {
        const files = Array.from(event.target.files);
        const readers = files.map((file) => {
          return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onloadend = () => {
                resolve({ name: file.name, file: reader.result });
              };
            reader.onerror = reject;
            reader.readAsDataURL(file);
          });
        });

        Promise.all(readers)
          .then((results) => setImages(results))
          .catch((error) => console.error("Error reading files: ", error));
    };
    
    return (
      <form 
        onSubmit={async ()=>{
        postReport(name,description,images);}
      }
        style = {{
            textAlign:"left",
            width:"30%",
            marginLeft:"35%",
        }}
      >
        <h2 style={{
            marginTop:"40px",
            marginBottom:"40px",
            textAlign:"center"
        }}
        >Crear Reporte</h2>
        <div className="form-group">
            <input 
                type="text" 
                value={name} 
                onChange={(e) => setName(e.target.value)} 
                placeholder="Nombre" 
                required 
                style={{
                    width:"100%",
                    marginBottom:"10px"
                }}
            />
        </div>
        <div className="form-group">
            <textarea 
                value={description} 
                onChange={(e) => setDescription(e.target.value)} 
                placeholder="Descripcion"
                style={{
                    width:"100%",
                    height:"135px"
                }}
            />
        </div>
        <div className="form-group">
            <input 
                type="file" 
                acceot=".jpeg, .png, .jpg"
                multiple 
                onChange={handleImageUpload} 
                required
                style={{
                    width:"100%",
                    borderColor:"transparent"
                }}
            />
        </div>
        <div className="form-group" style={{
            textAlign:"right"
        }}>
            <button type="submit" style={{
                borderColor:"transparent"
            }}> Crear reporte </button>
        </div>
      </form>
    );
  };

  export default ReportForm;
  