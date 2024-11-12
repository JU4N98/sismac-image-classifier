const BASE_URL = "http://localhost:8080";

const ENDPOINTS = {
    POST_REPORT : () => BASE_URL + "/report",
    LIST_REPORT : () => BASE_URL + "/report",
    GET_REPORT : (reportId) => BASE_URL + "/report/" + reportId,
};

export default ENDPOINTS;