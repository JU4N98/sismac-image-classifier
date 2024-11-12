import { useState, useEffect } from  "react";
import { useParams } from "react-router-dom";
import { getReport } from "../ApiManager/ApiManager";

export const ReportDetail = () => {
    const {reportId} = useParams();
    const [report, setReport] = useState({});

    useEffect(() => {
        getReport(reportId).
        then((response)=>{
            setReport(response.data);
            console.log(response.data);
        }).
        catch((error)=>{
            // console.log("XD");
            console.log(error);
        })
    },[]);

    return (
        <div style={{
            width:"80%",
            marginLeft:"10%",
            textAlign:"center"
        }}>
            <div style={galleryStyle}>
                {report.images?.map((image, index) => (
                    <div key={index} style={imageContainerStyle}>
                        <img
                        key={index}
                        src={image.file}
                        alt={`Gallery image ${index + 1}`}
                        style={imageStyle}
                        />
                        <div style={labelStyle}>
                            <strong>Nombre:</strong> {image.name}
                        </div>
                        <div style={{
                            ...labelStyle,
                            color: image.failure === "sin defectos" ? "green" : "red"
                        }}>
                            <strong>Fallo:</strong> {image.failure}
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