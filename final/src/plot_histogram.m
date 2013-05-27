function plot_histogram(data_file, image_file)
  % load histogram data
  load(data_file);
  
  % create figure, plot histogram data and save to requested file
  fh = figure;
  plot(histogram, 'lineWidth', 4);
  set(findall(fh, '-property', 'fontsize'), 'fontsize', 18);
  print(image_file, '-tight', '-color');
  close(fh);
end
