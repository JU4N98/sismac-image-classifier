import { useNavigate } from 'react-router-dom';


const NavBar = () => {
    const navigate = useNavigate();
    return (
        <nav className="navbar navbar-expand-lg navbar-light bg-light justify-content-center">
            <ul className="navbar-nav">
                <li className="nav-item">
                    <a className="nav-link" onClick={() => navigate('/createReport')}> Crear reporte </a>
                </li>
                <li className="nav-item" >
                    <a className="nav-link" onClick={() => navigate('/listReport')}> Ver reportes </a>
                </li>
            </ul>
        </nav>
    )
}

export default NavBar;
