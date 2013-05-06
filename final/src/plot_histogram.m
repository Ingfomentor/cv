function plot_histogram(data_file, image_file)
  % load histogram data
  load(data_file);
  
  % create figure, plot histogram data and save to requested file
  fh = figure;
  plot(histogram);
  print(image_file, '-tight', '-color');
  close(fh);
end
