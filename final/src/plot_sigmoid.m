function plot_sigmoid(param_file, image_file)
  % plots a sigmoid visualisation for given alpha and beta parameters

  % load alpha and beta
  load(param_file)

  % compute transformation of intensities 0:255 to stretched 0:255
  X = [0:255];
  Y = 255 ./ ( 1 + e .^ (-(X-double(beta))./double(alpha)) );
  
  fh = figure;
  plot(X, X, '-b', 'lineWidth', 4);
  hold on
  plot(X, Y, '-r', 'lineWidth', 4);
  plot([beta, beta], [0, 255], '--g', 'lineWidth', 4) 

  axis([0 255 0 255])
  legend('identical transform', 'stretching transform', 'beta')
  set(findall(fh, '-property', 'fontsize'), 'fontsize', 18);
  print(image_file, '-tight', '-color');
  close(fh);
end
