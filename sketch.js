// I LOVE WEBGPU

// TODO:
// - Model format (gltf?)
// - uniforms for camera matrix, object matrix
// - control
// - 4 samplers (albedo, roughness, metalness, normal)
// - deferred shading???
// - lights

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
  //canvas.style = "width: 400; height: 400;";
  
  const canvasContext = canvas.getContext( "webgpu" );

  canvasContext.configure( {
    device: device,
    format: navigator.gpu.getPreferredCanvasFormat(),
    alphaMode: "premultiplied",
  } );
  
  return canvasContext;
}


function createShaderModule( device, shaders )
{
  const shaderModule = device.createShaderModule( {code: shaders} );
  return shaderModule;
}


function bindVertexData( device, vertexDataArray )
{
  // Defined in CCW order -- see below: (frontFace: "ccw")
  const vertexBuffer = device.createBuffer( {
    size: vertexDataArray.byteLength, // make it big enough
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  } );
  
  device.queue.writeBuffer( vertexBuffer, 0, vertexDataArray, 0, vertexDataArray.length );
  

  return vertexBuffer;
}


function setupPipeline( device, shaderModule, vertexBufferDescriptor )
{
  const pipelineDescriptor = {
    vertex: {
      module: shaderModule,
      entryPoint: "vertex_main",
      buffers: vertexBufferDescriptor,
    },
    fragment: {
      module: shaderModule,
      entryPoint: "fragment_main",
      targets: [
        { format: navigator.gpu.getPreferredCanvasFormat() },
      ], 
    },
    primitive: {
      topology: "triangle-strip",
      frontFace: "ccw",
      cullMode: "back"
    },
    layout: "auto",
  };

  const renderPipeline = device.createRenderPipeline( pipelineDescriptor );
  
  return renderPipeline;
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