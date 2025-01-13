import ENDPOINTS from "./Endpoints";
import axios from "axios";

export const postReport = async (name, description, images) => {
    const url = ENDPOINTS.POST_REPORT();
    const payload =  {
        name,
        description,
        images
    }
    return await axios.post(
            url,
            payload,
            {
                headers: {"Content-Type": "application/json"}
            },
        );
}

export const getReports = async (page) => {
    const url = ENDPOINTS.LIST_REPORT(page);
    return await axios.get(
        url,
        {
            params: {
                page: page,
                page_size: 10,
            }
        }
    );
}

export const getReport = async (reportId) => {
    const url = ENDPOINTS.GET_REPORT(reportId);
    const response = await axios.get(
        url
    );
    return response;
}

export const putImage = async (imageId, image) => {
    const url = ENDPOINTS.UPDATE_IMAGE(imageId);
    return await axios.put(
        url,
        image,
        {
            headers: {"Content-Type": "application/json"}
        },
    );
}
