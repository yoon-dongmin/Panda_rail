// Generated by gencpp from file panda_move/HandRequest.msg
// DO NOT EDIT!


#ifndef PANDA_MOVE_MESSAGE_HANDREQUEST_H
#define PANDA_MOVE_MESSAGE_HANDREQUEST_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace panda_move
{
template <class ContainerAllocator>
struct HandRequest_
{
  typedef HandRequest_<ContainerAllocator> Type;

  HandRequest_()
    : grasp_state(false)
    , object_name()
    , grasp_size(0.0)  {
    }
  HandRequest_(const ContainerAllocator& _alloc)
    : grasp_state(false)
    , object_name(_alloc)
    , grasp_size(0.0)  {
  (void)_alloc;
    }



   typedef uint8_t _grasp_state_type;
  _grasp_state_type grasp_state;

   typedef std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other >  _object_name_type;
  _object_name_type object_name;

   typedef double _grasp_size_type;
  _grasp_size_type grasp_size;





  typedef boost::shared_ptr< ::panda_move::HandRequest_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::panda_move::HandRequest_<ContainerAllocator> const> ConstPtr;

}; // struct HandRequest_

typedef ::panda_move::HandRequest_<std::allocator<void> > HandRequest;

typedef boost::shared_ptr< ::panda_move::HandRequest > HandRequestPtr;
typedef boost::shared_ptr< ::panda_move::HandRequest const> HandRequestConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::panda_move::HandRequest_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::panda_move::HandRequest_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::panda_move::HandRequest_<ContainerAllocator1> & lhs, const ::panda_move::HandRequest_<ContainerAllocator2> & rhs)
{
  return lhs.grasp_state == rhs.grasp_state &&
    lhs.object_name == rhs.object_name &&
    lhs.grasp_size == rhs.grasp_size;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::panda_move::HandRequest_<ContainerAllocator1> & lhs, const ::panda_move::HandRequest_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace panda_move

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsFixedSize< ::panda_move::HandRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::panda_move::HandRequest_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::panda_move::HandRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::panda_move::HandRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::panda_move::HandRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::panda_move::HandRequest_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::panda_move::HandRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "3bca6d146ab3600c7e3fb19e63f1ad58";
  }

  static const char* value(const ::panda_move::HandRequest_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x3bca6d146ab3600cULL;
  static const uint64_t static_value2 = 0x7e3fb19e63f1ad58ULL;
};

template<class ContainerAllocator>
struct DataType< ::panda_move::HandRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "panda_move/HandRequest";
  }

  static const char* value(const ::panda_move::HandRequest_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::panda_move::HandRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "bool grasp_state\n"
"string object_name\n"
"float64 grasp_size\n"
;
  }

  static const char* value(const ::panda_move::HandRequest_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::panda_move::HandRequest_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.grasp_state);
      stream.next(m.object_name);
      stream.next(m.grasp_size);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct HandRequest_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::panda_move::HandRequest_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::panda_move::HandRequest_<ContainerAllocator>& v)
  {
    s << indent << "grasp_state: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.grasp_state);
    s << indent << "object_name: ";
    Printer<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > >::stream(s, indent + "  ", v.object_name);
    s << indent << "grasp_size: ";
    Printer<double>::stream(s, indent + "  ", v.grasp_size);
  }
};

} // namespace message_operations
} // namespace ros

#endif // PANDA_MOVE_MESSAGE_HANDREQUEST_H