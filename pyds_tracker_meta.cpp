#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "nvds_tracker_meta.h"

namespace py = pybind11;

PYBIND11_MODULE(pyds_tracker_meta, m) {
    m.doc() = "pybind11 wrapper to access Nvidia DeepStream NvDsPastFrame* classes";

    m.def("NvDsPastFrameObjBatch_cast",
        [](void *obj) { return (NvDsPastFrameObjBatch*)(obj); },
        "Cast given object/data to pyds_pastframemeta.NvDsPastFrameObjBatch",
        py::arg("obj"),
        py::return_value_policy::reference);

    m.def("NvDsPastFrameObjBatch_list",
        [](NvDsPastFrameObjBatch &self) {
            return py::make_iterator(self.list, self.list + self.numFilled);
        },
        py::keep_alive<0, 1>(),
        "Extracts an iterator of stream lists from the object batch.",
        py::arg("obj_batch"));

    m.def("NvDsPastFrameObjStream_list",
        [](NvDsPastFrameObjStream &self) {
            return py::make_iterator(self.list, self.list + self.numFilled);
        },
        py::keep_alive<0, 1>(),
        "Extracts an iterator of objects from an object stream.",
        py::arg("obj_stream"));

    m.def("NvDsPastFrameObjList_list",
        [](NvDsPastFrameObjList &self) { 
            return py::make_iterator(self.list, self.list + self.numObj); 
        },
        py::keep_alive<0, 1>(),
        "Extracts an iterator of past frame objects from this object list.",
        py::arg("obj_list"));

}
