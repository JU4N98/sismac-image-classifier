import { useState, useEffect } from  "react";
import { useParams } from "react-router-dom";
import { getReport, putImage } from "../ApiManager/ApiManager";

export const ReportDetail = () => {
    const {reportId} = useParams();
    const [report, setReport] = useState({});
    const [images, setImages] = useState([]);
    const [isEditing, setIsEditing] = useState(new Map());

    useEffect(() => {
        getReport(reportId)
        .then((response) => {
            setImages(response.data.images);
            setReport(response.data);
        }).catch((error) => {
            console.error(error);
        });
    }, [reportId]);
      
    useEffect(() => {
        if (images) {
            const newMap = new Map();
            images.forEach((image) => {
                newMap.set(image.id, false);
            });
            setIsEditing(newMap);
        }
    }, [images]);

    const handleClick = (id) => {
        const newMap = new Map(isEditing);
        newMap.set(id, !isEditing.get(id));
        setIsEditing(newMap);
    } 

    const handleChange = (id, failure) => {
        const updatedImage = {...images.find(image => image.id === id), failure: failure};
        delete updatedImage.id;
        
        putImage(id,updatedImage).then(
            setImages(prevImages =>
                prevImages.map(image =>
                image.id === id
                    ? {...image, failure:failure} 
                    : image
                )
            )
        )
    }

    return (
        <div style={{
            width:"80%",
            marginLeft:"10%",
            textAlign:"center"
        }}>
            <h2 style={{
                marginTop:"40px",
                marginBottom:"40px",
            }}>Reporte: {report.name}</h2>
            <p>
                Descripcion: {report.description}
            </p>
            <div style={galleryStyle}>
                {images?.map((image) => (
                    <div key={image.id} style={imageContainerStyle}>
                        <img
                            key={image.id}
                            src={image.file}
                            alt={image.name}
                            style={imageStyle}
                        />
                        <div style={labelStyle}>
                            <strong>Nombre:</strong> {image.name}
                        </div>
                        <div style={{...labelStyle,}}>{
                            isEditing.get(image.id) ?
                                <select
                                    onChange={(event)=>{
                                        handleChange(image.id,event.target.value);
                                        handleClick(image.id);
                                    }}
                                >
                                    <option value="">Elegir...</option>
                                    <option value="sin defectos">Sin defectos</option>
                                    <option value="sobrecarga en una fase">Sobrecarga en una fase</option>
                                    <option value="sobrecarga en dos fases">Sobrecarga en dos fases</option>
                                    <option value="sobrecarga en tres fases">Sobrecarga en tres fases</option>
                                </select>
                            :
                                <button 
                                    style={{border:"none",}} 
                                    onClick={()=>(handleClick(image.id))}>
                                    <strong style={{color: image.failure === "sin defectos" ? "green" : "red"}}>
                                        Fallo: {image.failure}
                                    </strong>
                                </button>
                        }
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}

const galleryStyle = {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))',
    gap: '20px',
    padding: '10px',
};
  
const imageContainerStyle = {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    textAlign: 'center',
};
  
const imageStyle = {
    width: '100%',
    height: 'auto',
    borderRadius: '8px',
    objectFit: 'cover',
};
  
const labelStyle = {
    marginTop: '5px',
    fontSize: '14px',
};
