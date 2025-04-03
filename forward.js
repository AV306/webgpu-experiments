const PI = 3.14159;
const TWO_PI = 6.28319;
const HALF_PI = 1.57080;

const depthBufferFormat = "depth24plus";

const cubeVertexData = new Float32Array( [
  -0.5, -0.5, -0.5,   0, 0, // Bottom back left
  0.5, -0.5, -0.5,    1, 0, // Bottom back right
  0.5, -0.5, 0.5,     1, 1, // Bottom front right
  -0.5, -0.5, 0.5,    0, 1, // Bottom front left
  
  -0.5, 0.5, -0.5,    0, 0, // Top back left
  0.5, 0.5, -0.5,     1, 0,
  0.5, 0.5, 0.5,      1, 1,
  -0.5, 0.5, 0.5,     0, 1
] );

// TODO: make these CCW
const cubeTriangleIndexData = new Uint16Array( [
  0, 1, 2, 0, 2, 3, // Bottom face
  4, 5, 6, 4, 6, 7, // Top face
  0, 4, 7, 0, 7, 3, // Left face
  1, 5, 6, 1, 6, 2, // right face
  0, 1, 5, 0, 5, 4, // Front face
  2, 6, 7, 2, 7, 3 // Back face
] );

const vertexBufferLayout = {
  attributes: [
    {
      // Position (vec3)
      shaderLocation: 0,
      offset: 0,
      format: "float32x3"
    },
    {
      // UV (vec2)
      shaderLocation: 1,
      offset: Float32Array.BYTES_PER_ELEMENT * 3,
      format: "float32x2"
    }
  ],
  //stepMode: "vertex",
  arrayStride: Float32Array.BYTES_PER_ELEMENT * 5, // 20 - vec3 (position) + vec2 (uv)
}

const cubeVertexCount = cubeVertexData.length;
const cubeIndexDataFormat = "uint16";

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

function bindModelVertexData( device, vertexData, indexData )
{
  const vertexBuffer = device.createBuffer( {
    //size: vertexData.length * vertexBufferLayout.arrayStride,
    size: vertexData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
  } );
  
  const indexBuffer = device.createBuffer( {
    size: indexData.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
  } );
  
  device.queue.writeBuffer( vertexBuffer, 0, vertexData, 0, vertexData.length );
  device.queue.writeBuffer( indexBuffer, 0, indexData, 0, indexData.length );
  
  return [vertexBuffer, indexBuffer];
}

function bindMatrixUniforms( device, pipeline, modelMatrix, viewMatrix, projectionMatrix )
{
  const modelMatrixBuffer = device.createBuffer( {
    size: modelMatrix.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  } );
  
  const viewMatrixBuffer = device.createBuffer( {
    size: viewMatrix.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  } );
  
  const projectionMatrixBuffer = device.createBuffer( {
    size: projectionMatrix.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  } );
  
  device.queue.writeBuffer( modelMatrixBuffer, 0, modelMatrix, 0, modelMatrix.length );
  device.queue.writeBuffer( viewMatrixBuffer, 0, viewMatrix, 0, viewMatrix.length );
  device.queue.writeBuffer( projectionMatrixBuffer, 0, projectionMatrix, 0, projectionMatrix.lengthv);

  const uniformBindGroup = device.createBindGroup( {
    layout: pipeline.getBindGroupLayout( 0 ),
    entries: [
      {
        binding: 0,
        resource: { buffer: modelMatrixBuffer }
      },
      {
        binding: 1,
        resource: { buffer: viewMatrixBuffer }
      },
      {
        binding: 2,
        resource: { buffer: projectionMatrixBuffer }
      }
    ]
  } );
  
  return [modelMatrixBuffer, viewMatrixBuffer, projectionMatrixBuffer, uniformBindGroup];
}

function createPipeline( device, presentationFormat, shaderModule )
{
  const pipeline = device.createRenderPipeline( {
    layout: 'auto',
    vertex: {
      module: shaderModule,
      buffers: [ vertexBufferLayout ],
    },
    fragment: {
      module: shaderModule,
      targets: [ { format: presentationFormat } ]
    },
    primitive: {
      topology: 'triangle-list',
      cullMode: "none", // Change to "back" after verifying that depth buffer works
    },
    depthStencil: {
      depthWriteEnabled: true,
      depthCompare: "less",
      format: depthBufferFormat,
    }
  });
  
  return pipeline;
}

function createDepthTexture( device, w=400, h=400 )
{
  const depthTexture = device.createTexture( {
    size: [w, h],
    format: depthBufferFormat,
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  } );
  return depthTexture;
}

function createRenderPassDescriptor( device, view, depthTexture, clearColor=[0, 0.5, 0.5, 1] )
{
  const renderPassDescriptor = {
    colorAttachments: [
      {
        view: view, // Assigned later

        clearValue: clearColor,
        loadOp: "clear",
        storeOp: "store"
      }
    ],
    depthStencilAttachment: {
      view: depthTexture.createView(),
      depthClearValue: 1.0,
      depthLoadOp: "clear",
      depthStoreOp: "store"
    }
  };
  
  return renderPassDescriptor;
}


function buildProjectionMatrix( horizontalFov, aspect, near=-1, far=-101 )
{
  return new Float32Array( [
    1/Math.tan( horizontalFov / 2 ), 0, 0, 0, // Column 0 (leftmost)
    0,aspect/Math.tan( horizontalFov / 2 ), 0, 0,
    0, 0, -(far + near)/(far - near), -1,
    0, 0, (2 * far * near)/(far - near), 0
  ] );
}


// ========== SETUP ==========
const device = await setupDevice();
const [canvasContext, aspect] = setupCanvasContext( device );
const [modelVertexBuffer, modelIndexBuffer] = bindModelVertexData( device, cubeVertexData, cubeIndexData );


// Annoyingly, matrices are COLUMN-MAJOR
const modelMatrix = new Float32Array( [
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1
] ); // Identity for now
const viewMatrix = new Float32Array( [
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1
] ); // Identity for now
const projectionMatrix = buildProjectionMatrix( HALF_PI, aspect );

// Bind matrix uniforms
const [modelMatrixBuffer, viewMatrixBuffer, projectionMatrixBuffer, uniformBindGroup] = bindMatrixUniforms( device, pipeline, modelMatrix, viewMatrix, projectionMatrix );

const pipeline = createPipeline( device, navigator.gpu.getPreferredCanvasFormat(), device.createShaderModule( {code: shaderText} ) );
const depthTexture = createDepthTexture( device );

const renderPassDescriptor = createRenderPassDescriptor( device, canvasContext.getCurrentTexture().createView(), depthTexture );


// ========== RENDER ==========
const commandEncoder = device.createCommandEncoder();
const passEncoder = commandEncoder.beginRenderPass( renderPassDescriptor );
passEncoder.setPipeline( pipeline );
passEncoder.setBindGroup( 0, uniformBindGroup );
passEncoder.setVertexBuffer( 0, modelVertexBuffer );
passEncoder.setIndexBuffer( modelIndexBuffer, modelIndexDataFormat );
passEncoder.drawIndexed( cubeIndexData.length );
passEncoder.end();

console.log( "here" );
device.queue.submit( [commandEncoder.finish()] );
