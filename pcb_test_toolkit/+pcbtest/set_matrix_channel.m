function set_matrix_channel(matrix, channel, role)
%SET_MATRIX_CHANNEL Route an injection or readout channel on the matrix.
    if ~ismember(channel, 1:15) || ~ismember(role, {'inject', 'readout'})
        error('pcbtest:MatrixChannel', 'Invalid matrix channel or role.');
    end
    if strcmp(role, 'inject')
        if channel <= 7
            write(matrix, 'U 8 3');
            write(matrix, sprintf('L 0 %d', channel));
        else
            write(matrix, 'L 8 3');
            write(matrix, sprintf('L 1 %d', channel - 8));
        end
    else
        if channel <= 7
            write(matrix, 'U 8 5');
            write(matrix, sprintf('L 6 %d', channel));
        else
            write(matrix, 'L 8 5');
            write(matrix, sprintf('L 7 %d', channel - 8));
        end
    end
end
