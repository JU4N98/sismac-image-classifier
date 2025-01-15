import './App.css';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import NavBar from './Navbar/NavBar';
import ReportForm from "./Report/CreateReport";
import ReportList from "./Report/ListReport";
import { ReportDetail } from "./Report/DetailReport";
import "bootstrap-icons/font/bootstrap-icons.css"

const App = () => {
  return (
    <Router>
      <NavBar/>
      <Routes>
        <Route path="/" element={<ReportForm/>} />
        <Route path="/createReport/" element={<ReportForm/>} />
        <Route path="/listReport" element={<ReportList/>} />
        <Route path="/listReport/:reportId" element={<ReportDetail/>}/>
      </Routes>
    </Router>
  );
}

export default App;
