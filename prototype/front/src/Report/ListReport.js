import { useState, useEffect } from "react";
import { useNavigate  } from "react-router-dom";
import { getReports } from "../ApiManager/ApiManager";

const ReportList = () => {
    const [page] = useState(1);
    const [reports, setReports] = useState([]);
    const navigate = useNavigate();

    useEffect(() => {
        getReports(page)
        .then((response)=>{
            setReports(response.data);
        })
        .catch((error)=>{
            console.log(error);
        })
    },[page]);

    const handleView = (reportId) => {
      navigate(`/listReport/${reportId}`);
    };

    return (
        <div style={{
            width:"30%",
            marginLeft:"35%",
            textAlign:"center"
        }}>
            <h2 style={{
                marginTop:"40px",
                marginBottom:"40px",
            }}>Lista de Reportes</h2>
            <table className="table table-hover table-bordered">
        <thead>
          <tr >
            <th styles={{
                width:"25%"
            }}>Fecha</th>
            <th>Nombre</th>
            <th style={{
                 width:"10%"
            }}></th>
          </tr>
        </thead>
        <tbody>
          {reports.map((report,index) => (
            <tr key={index}>
              <td style={{
                 textAlign:"left",
                 width:"25%"
              }}>{report.date}</td>
              <td style={{
                 textAlign:"left"
              }}>{report.name}</td>
              <td style={{
                width:"10%"    
              }}>
                <i onClick={()=>handleView(report.id)} className="bi bi-eye"></i>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
           
        </div>
    );
}

export default ReportList;