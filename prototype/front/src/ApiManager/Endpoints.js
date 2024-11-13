const {REACT_APP_BACKEND_URL} = process.env;
const BASE_URL = REACT_APP_BACKEND_URL;

const ENDPOINTS = {
    POST_REPORT : () => BASE_URL + "/report",
    LIST_REPORT : () => BASE_URL + "/report",
    GET_REPORT : (reportId) => BASE_URL + "/report/" + reportId,
};

export default ENDPOINTS;