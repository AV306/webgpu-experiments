const PI = Math.PI;
const TWO_PI = Math.PI * 2;
const HALF_PI = Math.PI / 2;

const depthBufferFormat = "depth24plus";

const cubeVertexData = new Float32Array( [
  -0.5, -0.5, 0.5,   0, 0, // Bottom back left
  0.5, -0.5, 0.5,    1, 0, // Bottom back right
  0.5, -0.5, -0.5,     1, 1, // Bottom front right
  -0.5, -0.5, -0.5,    0, 1, // Bottom front left
  
  -0.5, 0.5, 0.5,    0, 0, // Top back left
  0.5, 0.5, 0.5,     1, 0,
  0.5, 0.5, -0.5,      1, 1,
  -0.5, 0.5, -0.5,     0, 1
] );

// TODO: make these CCW
const cubeIndexData = new Uint16Array( [
  // 2, 1, 0, 3, 2, 0, // Bottom face
  // 4, 5, 6, 4, 6, 7, // Top face
  // 0, 4, 7, 0, 7, 3, // Left face
  // 1, 5, 6, 1, 6, 2, // right face
  // 0, 1, 5, 0, 5, 4, // Front face
  // 2, 6, 7, 2, 7, 3 // Back face
  2, 1, 0, 3, 2, 0,
  4, 5, 6, 4, 6, 7,
  3, 0, 4, 3, 4, 7,
  1, 2, 6, 1, 6, 5,
  0, 1, 5, 0, 5, 4,
  6, 2, 3, 6, 3, 7
] );

const squareVertexData = new Float32Array( [
  -0.5, -0.5, 0, 0, 0,
  0.5, -0.5, 0,  1, 0,
  0.5, 0.5, 0,   1, 1,
  -0.5, 0.5, 0,  0, 1
] );

const squareIndexData = new Uint16Array( [
  0, 1, 2,
  0, 2, 3
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
  
  device.addEventListener( "uncapturederror", event => console.log( event.error.message ) );
  
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
  
  //console.log( "hereewwe" );
  
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
    label: "projection matrix buffer",
    size: projectionMatrix.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  } );
  
  device.queue.writeBuffer( modelMatrixBuffer, 0, modelMatrix );
  device.queue.writeBuffer( viewMatrixBuffer, 0, viewMatrix );
  device.queue.writeBuffer( projectionMatrixBuffer, 0, projectionMatrix );

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
    layout: "auto",
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
      cullMode: "back", // Change to "back" after verifying that depth buffer works
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
        view: view,

        clearValue: clearColor,
        loadOp: "clear",
        storeOp: "store"
      }
    ],
    depthStencilAttachment: {
      view: depthTexture.createView(),
      depthClearValue: 1.0,
      depthLoadOp: "clear",
      depthStoreOp: "discard"
    }
  };
  
  return renderPassDescriptor;
}


function buildProjectionMatrix( horizontalFov, aspect, near=-1, far=-100 )
{
  const xScale = 1/Math.tan( horizontalFov / 2 );
  const yScale = aspect * xScale;
  const zScaleZ = (far + near)/(far - near);
  const zScaleW = 1; //(2 * far * near)/(near - far);
  /*return new Float32Array( [
    xScale, 0, 0, 0, // Column 0 (leftmost)
    0, yScale, 0, 0,
    0, 0, zScaleZ, -1,
    0, 0, zScaleW, 0
  ] );*/
  return new Float32Array( [
    xScale, 0, 0, 0, // Column 0 (leftmost)
    0, yScale, 0, 0,
    0, 0, 0, -1,
    0, 0, 0, 0
  ] );
}

// function buildProjectionMatrix( verticalFov, aspect, near=-1, far=-100 )
// {
//   /*return new Float32Array( [
//     1/(aspect * Math.tan( verticalFov / 2 )), 0, 0, 0, // Column 0 (leftmost)
//     0, 1/Math.tan( verticalFov / 2 ), 0, 0,
//     0, 0, -(far + near)/(far - near), -1,
//     0, 0, -(2 * far * near)/(far - near), 0
//   ] );*/
// }


// ========== SETUP ==========
const w = 400, h = 300;
const device = await setupDevice();
const [canvasContext, aspect] = setupCanvasContext( device, w, h );
const [modelVertexBuffer, modelIndexBuffer] = bindModelVertexData( device, cubeVertexData, cubeIndexData );

// Annoyingly, matrices are COLUMN-MAJOR
let angle = 0;
let modelMatrix = new Float32Array( [
  Math.cos( angle ), 0, Math.sin( angle ), 0,
  0, 1, 0, 0,
  -Math.sin( angle ), 0, Math.cos( angle ), 0,
  0, 0, 0, 1
] );
/*const modelMatrix = new Float32Array( [
  Math.cos( angle ), -Math.sin( angle ), 0, 0,
  Math.sin( angle ), Math.cos( angle ), 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1
] );*/

const viewMatrix = new Float32Array( [
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, -2, 1
] );

const projectionMatrix = buildProjectionMatrix( HALF_PI, aspect );

const shaderModule = device.createShaderModule( {code: document.getElementById( "shaders" ).innerText} );

const compilationInfo = await shaderModule.getCompilationInfo();
for ( const message of compilationInfo.messages )
  console.log( message );

const pipeline = createPipeline( device, navigator.gpu.getPreferredCanvasFormat(), shaderModule );

// Bind matrix uniforms
const [modelMatrixBuffer, viewMatrixBuffer, projectionMatrixBuffer, uniformBindGroup] = bindMatrixUniforms( device, pipeline, modelMatrix, viewMatrix, projectionMatrix );
const depthTexture = createDepthTexture( device, w, h );

const renderPassDescriptor = createRenderPassDescriptor( device, canvasContext.getCurrentTexture().createView(), depthTexture );


// ========== RENDER ==========
const sleep = (delay) => new Promise( (resolve) => setTimeout( resolve, delay ) );

function frame()
{
  modelMatrix[0] = modelMatrix[10] = Math.cos( angle );
  modelMatrix[2] = Math.sin( angle );
  modelMatrix[8] = -Math.sin( angle );
  
  device.queue.writeBuffer( modelMatrixBuffer, 0, modelMatrix );
  
  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginRenderPass( renderPassDescriptor );
  passEncoder.setPipeline( pipeline );
  passEncoder.setBindGroup( 0, uniformBindGroup );
  passEncoder.setVertexBuffer( 0, modelVertexBuffer );
  passEncoder.setIndexBuffer( modelIndexBuffer, cubeIndexDataFormat );
  passEncoder.drawIndexed( cubeIndexData.length );
  passEncoder.end();

  device.queue.submit( [commandEncoder.finish()] );

  angle += Math.PI / 20;
  setTimeout( frame, 100 );
}

frame();
