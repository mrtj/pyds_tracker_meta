#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "nvds_tracker_meta.h"

namespace py = pybind11;

PYBIND11_MODULE(pyds_tracker_meta, m) {
    m.doc() = "pybind11 wrapper to access Nvidia DeepStream NvDsPastFrame* classes";
    
    py::class_<NvDsPastFrameObjBatch>(m, "NvDsPastFrameObjBatch", "Batch of lists of buffered objects")
        
        // Casting and type check
        .def_static(
            "cast", 
            [](void *obj) { return (NvDsPastFrameObjBatch*)(obj); }, 
            "Cast given object/data to pyds_pastframemeta.NvDsPastFrameObjBatch", 
            py::arg("obj"),
            py::return_value_policy::reference)
        .def_static(
            "from_user_meta", 
            [](NvDsUserMeta *user_meta) {
                NvDsPastFrameObjBatch* obj = nullptr;
                if (user_meta && user_meta->base_meta.meta_type == NVDS_TRACKER_PAST_FRAME_META) {
                    obj = (NvDsPastFrameObjBatch *) (user_meta->user_meta_data);
                }
                return obj;
            }, 
            py::arg("user_meta"), 
            "If the data contained in the user meta is a NvDsPastFrameObjBatch instance, "
            "casts the data and returns it. Otherwise returns NULL.",
            py::return_value_policy::reference)
        .def_static(
            "is_in", 
            [](NvDsUserMeta *user_meta) {
                return (user_meta->base_meta.meta_type == NVDS_TRACKER_PAST_FRAME_META);
            }, 
            py::arg("user_meta"), 
            "Returns true if the user_meta contains a NvDsPastFrameObjBatch instance.")
        
        // Struct members
        .def_readonly("numAllocated", &NvDsPastFrameObjBatch::numAllocated, "Number of blocks allocated for the list.")
        .def_readonly("numFilled", &NvDsPastFrameObjBatch::numFilled, "Number of filled blocks in the list.")
    
        // list is implemented as a real python iterator
        .def_property_readonly(
            "list", 
            [](NvDsPastFrameObjBatch &self) { 
                return py::make_iterator(self.list, self.list + self.numFilled); 
            },
            py::keep_alive<0, 1>(),
            "Iterator of stream lists.")

        // convenience methods for subscription access, lenght, use the batch directly as an iterator
        .def(
            "__getitem__", 
            [](const NvDsPastFrameObjBatch &self, size_t i) {
                if (i >= self.numFilled) throw py::index_error();
                return self.list[i];
            })
        .def("__len__", [](const NvDsPastFrameObjBatch &self) { return self.numFilled; })
        .def("__iter__", 
            [](const NvDsPastFrameObjBatch &self) { 
                return py::make_iterator(self.list, self.list + self.numFilled); 
            },
            py::keep_alive<0, 1>());

    
    py::class_<NvDsPastFrameObjStream>(m, "NvDsPastFrameObjStream", "List of objects in each stream.")
        .def_readonly("streamID", &NvDsPastFrameObjStream::streamID, "Stream id the same as frame_meta->pad_index.")
        .def_readonly("surfaceStreamID", &NvDsPastFrameObjStream::surfaceStreamID, "Stream id used inside tracker plugin.")
        .def_readonly("numAllocated", &NvDsPastFrameObjStream::numAllocated, "Maximum number of objects allocated.")
        .def_readonly("numFilled", &NvDsPastFrameObjStream::numFilled, "Number of objects in this frame.")
    
        // list is implemented as a real python iterator
        .def_property_readonly(
            "list", 
            [](NvDsPastFrameObjStream &self) { return py::make_iterator(self.list, self.list + self.numFilled); },
            py::keep_alive<0, 1>(),
            "Iterator of objects inside this stream.")

        // convenience methods for subscription access, lenght, use the frame object stream directly as an iterator
        .def("__getitem__", 
            [](const NvDsPastFrameObjStream &self, size_t i) {
                if (i >= self.numFilled) throw py::index_error();
                return self.list[i];
            })
        .def("__len__", [](const NvDsPastFrameObjStream &self) { return self.numFilled; })
        .def("__iter__", 
            [](const NvDsPastFrameObjStream &self) { 
                return py::make_iterator(self.list, self.list + self.numFilled); 
            },
            py::keep_alive<0, 1>());


    py::class_<NvDsPastFrameObjList>(m, "NvDsPastFrameObjList", "One object in several past frames")
        .def_readonly("uniqueId", &NvDsPastFrameObjList::uniqueId, "Object tracking id.")
        .def_readonly("classId", &NvDsPastFrameObjList::classId, "Object class id.")
        .def_property_readonly(
            "objLabel", 
            [](NvDsPastFrameObjList *self) {
                return std::string(self->objLabel, MAX_LABEL_SIZE);
            }, 
            "An array of the string describing the object class.")
        .def_readonly("numObj", &NvDsPastFrameObjList::numObj, "Number of frames this object appreared in the past.")

        // list is implemented as a real python iterator
        .def_property_readonly(
            "list", 
            [](NvDsPastFrameObjList &self) { return py::make_iterator(self.list, self.list + self.numObj); },
            py::keep_alive<0, 1>(),
            "Iterator of past frame info of this object.")

        // convenience methods for subscription access, lenght, use the frame object list directly as an iterator
        .def("__getitem__", 
            [](const NvDsPastFrameObjList &self, size_t i) {
                if (i >= self.numObj) throw py::index_error();
                return self.list[i];
            })
        .def("__len__", [](const NvDsPastFrameObjList &self) { return self.numObj; })
        .def("__iter__", 
            [](const NvDsPastFrameObjList &self) { 
                return py::make_iterator(self.list, self.list + self.numObj); 
            },
            py::keep_alive<0, 1>());


    py::class_<NvDsPastFrameObj>(m, "NvDsPastFrameObj")
        .def_readonly("frameNum", &NvDsPastFrameObj::frameNum)
        .def_readonly("tBbox", &NvDsPastFrameObj::tBbox)
        .def_readonly("confidence", &NvDsPastFrameObj::confidence)
        .def_readonly("age", &NvDsPastFrameObj::age);

}
