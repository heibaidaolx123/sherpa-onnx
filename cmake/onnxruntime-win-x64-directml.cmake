# Copyright (c)  2022-2023  Xiaomi Corporation
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_VS_PLATFORM_NAME: ${CMAKE_VS_PLATFORM_NAME}")
message(STATUS "Using Sherpa ONNX with DirectML")

if(NOT CMAKE_SYSTEM_NAME STREQUAL Windows)
  message(FATAL_ERROR "This file is for Windows only. Given: ${CMAKE_SYSTEM_NAME}")
endif()

if(NOT (CMAKE_VS_PLATFORM_NAME STREQUAL X64 OR CMAKE_VS_PLATFORM_NAME STREQUAL x64))
  message(FATAL_ERROR "This file is for Windows x64 only. Given: ${CMAKE_VS_PLATFORM_NAME}")
endif()

if(NOT BUILD_SHARED_LIBS)
  message(FATAL_ERROR "This file is for building shared libraries. BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")
endif()

if(NOT SHERPA_ONNX_ENABLE_DIRECTML)
  message(FATAL_ERROR "This file is for DirectML. Given SHERPA_ONNX_ENABLE_DIRECTML: ${SHERPA_ONNX_ENABLE_DIRECTML}")
endif()

if(location_onnxruntime_header_dir AND location_onnxruntime_lib)
    message("Use preinstall onnxruntime with directml: ${location_onnxruntime_lib}")
else()
    # set(VERSION_ONNXRUNTIME "1.14.1")
    # set(onnxruntime_URL  "https://globalcdn.nuget.org/packages/microsoft.ml.onnxruntime.directml.1.14.1.nupkg")
    # set(onnxruntime_URL2 "https://hf-mirror.com/csukuangfj/sherpa-onnx-cmake-deps/resolve/main/microsoft.ml.onnxruntime.directml.1.14.1.nupkg")
    # set(onnxruntime_HASH "SHA256=c8ae7623385b19cd5de968d0df5383e13b97d1b3a6771c9177eac15b56013a5a")

    # set(VERSION_ONNXRUNTIME "1.17.1")
    # set(onnxruntime_URL  "https://globalcdn.nuget.org/packages/microsoft.ml.onnxruntime.directml.1.17.1.nupkg")
    # set(onnxruntime_URL2 "https://hf-mirror.com/csukuangfj/sherpa-onnx-cmake-deps/resolve/main/microsoft.ml.onnxruntime.directml.1.17.1.nupkg")
    # set(onnxruntime_HASH "SHA256=834e9f02a348be0ae0fdf0e71df59661b64072e6f89fd6da19bcf74765f6574e")

    # set(VERSION_ONNXRUNTIME "1.22.0")
    # set(onnxruntime_URL  "https://globalcdn.nuget.org/packages/microsoft.ml.onnxruntime.directml.1.22.0.nupkg")
    # set(onnxruntime_URL2 "https://hf-mirror.com/csukuangfj/sherpa-onnx-cmake-deps/resolve/main/microsoft.ml.onnxruntime.directml.1.22.0.nupkg")
    # set(onnxruntime_HASH "SHA256=29f9872d786236b79aa83f94482f3a17c14297e4833768d6d0ed4883ee732e60")

    set(VERSION_ONNXRUNTIME "1.22.1")
    set(onnxruntime_URL  "https://globalcdn.nuget.org/packages/microsoft.ml.onnxruntime.directml.1.22.1.nupkg")
    set(onnxruntime_URL2 "https://hf-mirror.com/csukuangfj/sherpa-onnx-cmake-deps/resolve/main/microsoft.ml.onnxruntime.directml.1.22.1.nupkg")
    set(onnxruntime_HASH "SHA256=210823f9932f81f222e2d05d04531d948a4d75dbf527dfb94195ad61371ab728")

    message(STATUS "VERSION_ONNXRUNTIME: ${VERSION_ONNXRUNTIME}")
    # If you don't have access to the Internet,
    # please download onnxruntime to one of the following locations.
    # You can add more if you want.
    set(possible_file_locations
        $ENV{HOME}/Downloads/microsoft.ml.onnxruntime.directml.${VERSION_ONNXRUNTIME}.nupkg
        $ENV{HOME}/Downloads/microsoft.ml.onnxruntime.directml.${VERSION_ONNXRUNTIME}.zip
        ${PROJECT_SOURCE_DIR}/microsoft.ml.onnxruntime.directml.${VERSION_ONNXRUNTIME}.nupkg
        ${PROJECT_BINARY_DIR}/microsoft.ml.onnxruntime.directml.${VERSION_ONNXRUNTIME}.nupkg
        /tmp/microsoft.ml.onnxruntime.directml.${VERSION_ONNXRUNTIME}.nupkg
    )

    foreach(f IN LISTS possible_file_locations)
      if(EXISTS ${f})
        set(onnxruntime_URL  "${f}")
        file(TO_CMAKE_PATH "${onnxruntime_URL}" onnxruntime_URL)
        message(STATUS "Found local downloaded onnxruntime: ${onnxruntime_URL}")
        set(onnxruntime_URL2)
        break()
      endif()
    endforeach()

    FetchContent_Declare(onnxruntime
      URL
        ${onnxruntime_URL}
        ${onnxruntime_URL2}
      URL_HASH          ${onnxruntime_HASH}
    )

    FetchContent_GetProperties(onnxruntime)
    if(NOT onnxruntime_POPULATED)
      message(STATUS "Downloading onnxruntime from ${onnxruntime_URL}")
      FetchContent_Populate(onnxruntime)
    endif()
    message(STATUS "onnxruntime is downloaded to ${onnxruntime_SOURCE_DIR}")

    find_library(location_onnxruntime onnxruntime
      PATHS
      "${onnxruntime_SOURCE_DIR}/runtimes/win-x64/native"
      NO_CMAKE_SYSTEM_PATH
    )

    message(STATUS "location_onnxruntime: ${location_onnxruntime}")

    add_library(onnxruntime SHARED IMPORTED)

    set_target_properties(onnxruntime PROPERTIES
      IMPORTED_LOCATION ${location_onnxruntime}
      INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_SOURCE_DIR}/build/native/include"
    )

    set_property(TARGET onnxruntime
      PROPERTY
        IMPORTED_IMPLIB "${onnxruntime_SOURCE_DIR}/runtimes/win-x64/native/onnxruntime.lib"
    )

    file(COPY ${onnxruntime_SOURCE_DIR}/runtimes/win-x64/native/onnxruntime.dll
      DESTINATION
        ${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}
    )

    file(GLOB onnxruntime_lib_files "${onnxruntime_SOURCE_DIR}/runtimes/win-x64/native/onnxruntime.*")

    message(STATUS "onnxruntime lib files: ${onnxruntime_lib_files}")

    if(SHERPA_ONNX_ENABLE_PYTHON)
      install(FILES ${onnxruntime_lib_files} DESTINATION ..)
    else()
      install(FILES ${onnxruntime_lib_files} DESTINATION lib)
    endif()

    install(FILES ${onnxruntime_lib_files} DESTINATION bin)

endif()

# Setup DirectML

set(directml_URL "https://www.nuget.org/api/v2/package/Microsoft.AI.DirectML/1.15.4")
set(directml_HASH "SHA256=4e7cb7ddce8cf837a7a75dc029209b520ca0101470fcdf275c1f49736a3615b9")

set(possible_directml_file_locations
    $ENV{HOME}/Downloads/Microsoft.AI.DirectML.1.15.4.nupkg
    $ENV{HOME}/Downloads/Microsoft.AI.DirectML.1.15.4.zip
    ${PROJECT_SOURCE_DIR}/Microsoft.AI.DirectML.1.15.4.nupkg
    ${PROJECT_BINARY_DIR}/Microsoft.AI.DirectML.1.15.4.nupkg
    /tmp/Microsoft.AI.DirectML.1.15.4.nupkg
)

foreach(f IN LISTS possible_directml_file_locations)
  if(EXISTS ${f})
    set(directml_URL  "${f}")
    file(TO_CMAKE_PATH "${directml_URL}" directml_URL)
    message(STATUS "Found local downloaded DirectML: ${directml_URL}")
    break()
  endif()
endforeach()

FetchContent_Declare(directml
  URL
    ${directml_URL}
  URL_HASH ${directml_HASH}
)

FetchContent_GetProperties(directml)
if(NOT directml_POPULATED)
  message(STATUS "Downloading DirectML from ${directml_URL}")
  FetchContent_Populate(directml)
endif()
message(STATUS "DirectML is downloaded to ${directml_SOURCE_DIR}")

find_library(location_directml DirectML
  PATHS
  "${directml_SOURCE_DIR}/bin/x64-win"
  NO_CMAKE_SYSTEM_PATH
)

message(STATUS "location_directml: ${location_directml}")

add_library(directml SHARED IMPORTED)

set_target_properties(directml PROPERTIES
  IMPORTED_LOCATION ${location_directml}
  INTERFACE_INCLUDE_DIRECTORIES "${directml_SOURCE_DIR}/bin/x64-win"
)

set_property(TARGET directml
  PROPERTY
    IMPORTED_IMPLIB "${directml_SOURCE_DIR}/bin/x64-win/DirectML.lib"
)

file(COPY ${directml_SOURCE_DIR}/bin/x64-win/DirectML.dll
  DESTINATION
    ${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}
)

file(GLOB directml_lib_files "${directml_SOURCE_DIR}/bin/x64-win/DirectML.*")

message(STATUS "DirectML lib files: ${directml_lib_files}")

install(FILES ${directml_lib_files} DESTINATION lib)
install(FILES ${directml_lib_files} DESTINATION bin)
