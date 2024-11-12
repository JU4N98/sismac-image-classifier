import { useNavigate } from 'react-router-dom';


const NavBar = () => {
    const navigate = useNavigate();
    return (
        <nav class="navbar navbar-expand-lg navbar-light bg-light justify-content-center">
            <ul class="navbar-nav">
                <li class="nav-item">
                    <a class="nav-link" onClick={() => navigate('/createReport')}> Crear reporte </a>
                </li>
                <li class="nav-item" >
                    <a class="nav-link" onClick={() => navigate('/listReport')}> Ver reportes </a>
                </li>
            </ul>
        </nav>
    )
}

export default NavBar;
