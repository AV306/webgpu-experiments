// I LOVE WEBGPU

import {
    cubeVertexDataBuffer,
    cubeVertexBufferDescriptor,
    cubePrimitiveFormat,
    cubeVertexStrideBytes,
    cubeVertexSize,
    cubeUVOffset,
    cubePositionOffset,
    cubeVertexCount,
} from "cube.js";

// TODO:
// - uniform buffers

// Load shaders, defined in HTML
const shaders = document.getElementById( "shaders" ).innerText;


async function setupDevice()
{
    const adapter = await navigator.gpu?.requestAdapter();
    const device = await adapter?.requestDevice();
    if ( !device )
        throw Error( "browser doesn't support webgpu :(" );
    
    return device;
}


function setupCanvasContext( device, w=400, h=400 )
{
    const canvas = document.getElementById( "canvas" );
    canvas.width = w;
    canvas.height = h;
    const aspect = canvas.width / canvas.height;
    //canvas.style = "width: 400; height: 400;";
    
    const canvasContext = canvas.getContext( "webgpu" );
    
    canvasContext.configure( {
        device: device,
        format: navigator.gpu.getPreferredCanvasFormat(),
        alphaMode: "premultiplied",
    } );
    
    return [canvasContext, aspect];
}


function createShaderModule( device, shaders )
{
    const shaderModule = device.createShaderModule( {code: shaders} );
    return shaderModule;
}

function bindModelVertexData( device, modelVertexDataArray )
{
    const modelVertexBuffer = device.createBuffer( {
        size: modelVertexDataArray.length * cubeVertexStrideBytes,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
    } );
    
    // Copy vertex data array to GPU buffer :D
    device.queue.writeBuffer( vertexBuffer, 0, modelVertexDataArray, 0, modelVertexDataArray.length );
    
    /*{
    const mapping = new Float32Array( vertexBuffer.getMappedRange() );
    for ( let i = 0; i < mesh.positions.length; ++i )
    {
    mapping.set( mesh.positions[i], kVertexStride * i);
    mapping.set( mesh.normals[i], kVertexStride * i + 3);
    mapping.set( mesh.uvs[i], kVertexStride * i + 6);
    }
    vertexBuffer.unmap();
    }*/
    
    return vertexBuffer;
}


function bindLightsData( device )
{
    
}


/**
* Create a G-buffer target
* @param {*} device the GPU device
* @param {*} width the width of the G-buffer
* @param {*} height the height of the G-buffer
* @param {*} format the data format of the G-buffer (rgba16float, depth24plus, ...)
*/
function createGBuffer( device, width, height, format )
{
    const gbufferTexture = device.createTexture( {
        size: [width, height],
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        format: format
    } );
    
    return gbufferTexture;
}


/**
* Create a pipeline specifically for G-buffer writing
* @param {*} device the GPU device
* @param {*} vertexBufferDescriptors the list of vertex buffer descriptors that will be used in the pipeline
* @param {*} gBufferWriteShaderModule the shader module (with both vertex and fragment shaders) that will write to all G-buffers
* @param {*} gBufferTargetFormats the format objects ({format: rgba16float}...) of the G-buffers
*/
function createGBufferWritePipeline( device, vertexBufferDescriptors, globalPrimitiveFormat, gBufferWriteShaderModule, gBufferTargetFormats )
{
    // FIXME: depth + stencil is hardcoded
    const pipeline = device.createRenderPipeline( {
        layout: "auto",
        vertex: {
            module: gBufferWriteShaderModule,
            buffers: vertexBufferDescriptors // TODO: find out what happens when multiple descriptors are used
        },
        fragment: {
            module: gBufferWriteShaderModule,
            targets: gBufferTargetFormats
        },
        depthStencil: {
            depthWriteEnabled: true,
            depthCompare: 'less',
            format: 'depth24plus',
        },
        globalPrimitiveFormat
    } );
    
    return pipeline;
}

function createGBufferWritePassDescriptor( device, gBufferTextureViews )
{
    const passDescriptor = {
        colorAttachments: [
            {
                // Depth (FIXME: ???)
                view: gBufferTextureViews[0],
                clearValue: [0.0, 0.0, 1.0, 1.0],
                loadOp: 'clear',
                storeOp: 'store',
            },
            {
                // Albedo
                view: gBufferTextureViews[1],
                clearValue: [0, 0, 0, 1],
                loadOp: 'clear',
                storeOp: 'store',
            },
        ],
        depthStencilAttachment: {
            view: depthTexture.createView(),
            depthClearValue: 1.0,
            depthLoadOp: 'clear',
            depthStoreOp: 'store',
        },
    };
    
    return passDescriptor;
}


/**
* Create a texture bind group layout specifically for the G-buffers
* @param {*} device 
*/
function createGBufferBindGroupLayout( device )
{
    const layout = device.createBindGroupLayout( {
        entries: [
            {
                // Albedo
                binding: 0,
                visibility: GPUShaderStage.FRAGMENT,
                texture: {
                    sampleType: 'unfilterable-float',
                }
            },
            {
                // Normal
                binding: 1,
                visibility: GPUShaderStage.FRAGMENT,
                texture: {
                    sampleType: 'unfilterable-float',
                }
            },
            {
                // UV
                binding: 2,
                visibility: GPUShaderStage.FRAGMENT,
                texture: {
                    sampleType: 'unfilterable-float',
                }
            }
        ]
    } );
    
    return layout;
}


function createCompositingPipeline( device, gBufferBindGroupLayout, lightsBufferBindGroupLayout, globalPrimitiveFormat, compositingShaderModule, globalPresentationFormat )
{
    const pipeline = device.createRenderPipeline( {
        layout: device.createPipelineLayout( {
            bindGroupLayouts: [
                gBufferBindGroupLayout,
                lightsBufferBindGroupLayout,
            ],
        } ),
        vertex: {
            module: compositingShaderModule,
        },
        fragment: {
            module: compositingShaderModule,
            targets: [
                { format: globalPresentationFormat }
            ]
        },
        globalPrimitiveFormat
    } );
    
    return pipeline;
}


function createCompositingPassDescriptor( device )
{
    const passDescriptor = {
        colorAttachments: [
          {
            // View is acquired and set in render loop
            view: undefined,
            clearValue: [0, 0, 0, 1],
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
      };
 // TODO: stopped here
    return passDescriptor;
}


function setupUniforms( device )
{
    
}


function render( device, canvasContext, renderPipeline, vertexBuffers )
{
    const commandEncoder = device.createCommandEncoder();
    
    const clearColor = { r: 0.0, g: 0.5, b: 1.0, a: 1.0 };
    
    const renderPassDescriptor = {
        colorAttachments: [
            {
                clearValue: clearColor,
                loadOp: "clear",
                storeOp: "store",
                view: canvasContext.getCurrentTexture().createView(),
            },
        ],
    };
    
    const passEncoder = commandEncoder.beginRenderPass( renderPassDescriptor );
    
    passEncoder.setPipeline( renderPipeline );
    for ( const vertexData of vertexBuffers )
        {
        passEncoder.setVertexBuffer( 0, vertexData );
        passEncoder.draw( 6 );
    }
    
    passEncoder.end();
    device.queue.submit( [commandEncoder.finish()] );
}



async function run()
{
    const cubeVertices = new Float32Array( [
        // pos (vec4), texcoord (vec2)
        -0.5, 0.5, 0.5, 1,
        0, 0,
        
        -0.5, -0.5, 0.5, 1,
        1, 0,
        
        0.5, 0.5, 0.5, 1,
        0, 1,
        
        0.5, -0.5, 0.5, 1,
        1, 1,
        
        
        0.5, 0.5, -0.5, 1,
        0, 0,
        
        0.5, -0.5, -0.5, 1,
        1, 0,
    ] );
    
    // Create descriptor for vertex buffer format
    // All vertex buffers must follow this format
    const cubeVertexBufferDescriptor = [
        {
            attributes: [
                {
                    shaderLocation: 0, // position for shader
                    offset: 0,
                    format: "float32x4", // 4x 4-byte (32-bit) floats
                },
                {
                    shaderLocation: 1, // texcoord for shader
                    offset: 16, // The texcoords are the next set of float32x4
                    format: "float32x2", // 2x 4-byte floats
                },
            ],
            arrayStride: 24, // Each block of (position, colour) is 24 bytes
            stepMode: "vertex",
        },
    ];
    
    const device = await setupDevice();
    const canvasContext = setupCanvasContext( device );
    const shaderModule = createShaderModule( device, shaders );
    const vertexBuffer = bindVertexData( device, cubeVertices );
    const renderPipeline = setupPipeline( device, shaderModule, cubeVertexBufferDescriptor );
    
    render( device, canvasContext, renderPipeline, [vertexBuffer] );
}
run();